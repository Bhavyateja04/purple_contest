"""
Real-time store metrics computation.

Computes:
- Unique visitors (excluding staff)
- Conversion rate (correlated with POS data)
- Average dwell per zone
- Current queue depth
- Abandonment rate
"""

import csv
import json
import logging
import os
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import StoreMetrics, ZoneDwellMetric
from app.database import EventRecord
from datetime import date
logger = logging.getLogger(__name__)

# Path to POS data (relative to project root or via env)
POS_CSV = os.getenv(
    "POS_CSV_PATH",
    str(Path(__file__).parent.parent.parent / "Brigade_Bangalore_10_April_26 (1)bc6219c.csv"),
)

# Billing zone IDs (used for conversion correlation)
BILLING_ZONES = {"CASH_COUNTER"}

# POS correlation window: visitor in billing zone within N minutes before transaction
CONVERSION_WINDOW_MIN = 5.0


def _load_pos_transactions(store_id: str, target_date: Optional[date] = None) -> List[dict]:
    """Load POS transactions for a store on a specific date."""
    transactions = []
    # print("POS_CSV PATH =", POS_CSV)
    # print("FILE EXISTS =", os.path.exists(POS_CSV))
    if not os.path.exists(POS_CSV):
        logger.warning(f"POS CSV not found at {POS_CSV}")
        return transactions

    with open(POS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                order_date = datetime.strptime(row["order_date"], "%d-%m-%Y").date()
                if target_date and order_date != target_date:
                    continue
                order_dt = datetime.strptime(
                    f"{row['order_date']} {row['order_time']}",
                    "%d-%m-%Y %H:%M:%S",
                ).replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
                transactions.append(
                    {
                        "order_id": row["order_id"],
                        "timestamp": order_dt,
                        "amount": float(row.get("total_amount") or 0),
                        "customer_phone": row.get("customer_number", ""),
                    }
                )
            except Exception:
                continue
    logger.debug(f"Loaded {len(transactions)} POS transactions for {store_id}")
    return transactions


def _count_converted_visitors(
    billing_events: List[dict], pos_transactions: List[dict]
) -> int:
    """
    Count unique visitor_ids who were in a billing zone in the N-minute window
    before a POS transaction.
    """
    converted_visitors: set = set()
    window_sec = CONVERSION_WINDOW_MIN * 60

    for txn in pos_transactions:
        txn_ts = txn["timestamp"]
        for evt in billing_events:
            evt_ts = evt["timestamp"]
            # SQLite may return timestamps as strings
            if isinstance(evt_ts, str):
                try:
                    evt_ts = datetime.fromisoformat(evt_ts.replace("Z", "+00:00"))
                except Exception:
                    continue
            if evt_ts.tzinfo is None:
                evt_ts = evt_ts.replace(tzinfo=timezone.utc)
            delta = (txn_ts - evt_ts).total_seconds()
            if 0 <= delta <= window_sec:
                converted_visitors.add(evt["visitor_id"])

    return len(converted_visitors)


def compute_metrics(store_id: str, db: Session, target_date: Optional[date] = None) -> StoreMetrics:
    """Compute real-time store metrics for a given store and date."""
    if target_date is None:
        target_date = date(2026, 4, 10)

    date_str = target_date.isoformat()
    date_start = datetime.combine(target_date, datetime.min.time()).replace(
        tzinfo=timezone.utc
    )
    date_end = date_start + timedelta(days=1)

    # --- Unique visitors (ENTRY events, not staff) ---
    entry_rows = db.execute(
        text("""
            SELECT DISTINCT visitor_id
            FROM events
            WHERE store_id = :sid
              AND event_type IN ('ZONE_ENTER', 'ENTRY', 'REENTRY')
              AND is_staff = 0
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchall()
    unique_visitors = len(entry_rows)

    # --- Zone dwell metrics ---
    dwell_rows = db.execute(
        text("""
            SELECT zone_id, AVG(dwell_ms), COUNT(*)
            FROM events
            WHERE store_id = :sid
              AND event_type IN ('ZONE_DWELL', 'ZONE_EXIT')
              AND is_staff = 0
              AND zone_id IS NOT NULL
              AND timestamp >= :start
              AND timestamp < :end
            GROUP BY zone_id
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchall()

    zone_metrics = [
        ZoneDwellMetric(
            zone_id=row[0],
            avg_dwell_sec=round((row[1] or 0) / 1000.0, 1),
            visit_count=row[2],
        )
        for row in dwell_rows
        if row[0]
    ]

    # --- Queue depth (most recent billing event) ---
    queue_row = db.execute(
        text("""
            SELECT queue_depth
            FROM events
            WHERE store_id = :sid
              AND event_type = 'BILLING_QUEUE_JOIN'
              AND timestamp >= :start
            ORDER BY timestamp DESC
            LIMIT 1
        """),
        {"sid": store_id, "start": date_start},
    ).fetchone()
    queue_depth = int(queue_row[0]) if queue_row and queue_row[0] is not None else 0

    # --- Abandonment rate ---
    abandon_rows = db.execute(
        text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id = :sid
              AND event_type = 'BILLING_QUEUE_ABANDON'
              AND is_staff = 0
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchone()
    total_billing_join = db.execute(
        text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id = :sid
              AND event_type = 'BILLING_QUEUE_JOIN'
              AND is_staff = 0
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchone()

    abandon_count = int(abandon_rows[0]) if abandon_rows else 0
    billing_join_count = int(total_billing_join[0]) if total_billing_join else 0
    abandonment_rate = (
        round(abandon_count / billing_join_count, 3) if billing_join_count > 0 else 0.0
    )

    # --- Conversion rate (POS correlation) ---
    billing_events_rows = db.execute(
        text("""
            SELECT visitor_id, timestamp
            FROM events
            WHERE store_id = :sid
              AND event_type IN ('ZONE_ENTER', 'ZONE_DWELL')
              AND zone_id IN ('CASH_COUNTER')
              AND is_staff = 0
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchall()

    billing_events = [
        {"visitor_id": row[0], "timestamp": row[1]} for row in billing_events_rows
    ]

    # Attempt IST-adjusted date for POS lookup
    ist_date = date(2026, 4, 10)  # POS data is for this date
    if target_date == ist_date or target_date == date.today():
        pos_txns = _load_pos_transactions(store_id, ist_date)
    else:
        pos_txns = _load_pos_transactions(store_id, target_date)

    total_transactions = len(set(t["order_id"] for t in pos_txns))
    avg_basket = (
        sum(t["amount"] for t in pos_txns) / total_transactions
        if total_transactions > 0
        else 0.0
    )

    converted = _count_converted_visitors(billing_events, pos_txns)
    conversion_rate = round(converted / unique_visitors, 3) if unique_visitors > 0 else 0.0

    return StoreMetrics(
        store_id=store_id,
        date=date_str,
        unique_visitors=unique_visitors,
        conversion_rate=conversion_rate,
        avg_basket_value_inr=round(avg_basket, 2),
        avg_dwell_per_zone=zone_metrics,
        queue_depth_now=queue_depth,
        abandonment_rate=abandonment_rate,
        total_transactions=total_transactions,
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
