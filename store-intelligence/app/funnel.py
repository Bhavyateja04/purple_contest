"""
Conversion funnel computation.

Session is the unit of analysis (not raw events).
Funnel stages:
  1. Entry      — visitor crossed the threshold
  2. Zone Visit — visited at least one named zone
  3. Billing    — entered billing zone
  4. Purchase   — a POS transaction correlates to this session

Re-entries do NOT double-count a visitor — each visitor_id is counted once.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import StoreFunnel, FunnelStage
from app.metrics import _load_pos_transactions, _count_converted_visitors, BILLING_ZONES

logger = logging.getLogger(__name__)


def compute_funnel(store_id: str, db: Session, target_date: Optional[date] = None) -> StoreFunnel:
    """Build the conversion funnel for a store on a given date."""
    if target_date is None:
        target_date = date(2026, 4, 10)

    date_start = datetime.combine(target_date, datetime.min.time()).replace(
        tzinfo=timezone.utc
    )
    date_end = date_start + timedelta(days=1)
    date_str = target_date.isoformat()

    # Stage 1: Unique entrants (ENTRY events, deduplicated by visitor_id, no staff)
    entry_rows = db.execute(
        text("""
            SELECT DISTINCT visitor_id
            FROM events
            WHERE store_id = :sid
              AND event_type = 'ZONE_ENTER'
              AND zone_id = 'ENTRY'
              AND is_staff = 0
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchall()
    entered_visitors = {row[0] for row in entry_rows}
    n_entry = len(entered_visitors)

    # Stage 2: Visited at least one named zone (ZONE_ENTER events, not staff)
    zone_rows = db.execute(
        text("""
            SELECT DISTINCT visitor_id
            FROM events
            WHERE store_id = :sid
              AND event_type = 'ZONE_ENTER'
              AND is_staff = 0
              AND zone_id NOT IN ('ENTRY', 'EXIT')
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchall()
    zone_visitors = {row[0] for row in zone_rows} & entered_visitors
    n_zone = len(zone_visitors)

    # Stage 3: Reached billing zone
    billing_zones_str = ", ".join(f"'{z}'" for z in BILLING_ZONES)
    billing_rows = db.execute(
        text(f"""
            SELECT DISTINCT visitor_id
            FROM events
            WHERE store_id = :sid
              AND event_type IN ('ZONE_ENTER', 'BILLING_QUEUE_JOIN')
              AND is_staff = 0
              AND zone_id IN ({billing_zones_str})
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchall()
    billing_visitors = {row[0] for row in billing_rows} & entered_visitors
    n_billing = len(billing_visitors)

    # Stage 4: Made a purchase (POS correlation)
    billing_events_rows = db.execute(
        text(f"""
            SELECT visitor_id, timestamp
            FROM events
            WHERE store_id = :sid
              AND zone_id IN ({billing_zones_str})
              AND is_staff = 0
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchall()
    billing_event_dicts = []
    for row in billing_events_rows:
        ts = row[1]
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
        billing_event_dicts.append({"visitor_id": row[0], "timestamp": ts})

    pos_txns = _load_pos_transactions(store_id, target_date)
    n_purchase = _count_converted_visitors(billing_event_dicts, pos_txns)
    # Cap at billing visitors
    n_purchase = min(n_purchase, n_billing)

    # Compute drop-off percentages
    def drop_off(current: int, previous: int) -> float:
        if previous == 0:
            return 0.0
        return round((1.0 - current / previous) * 100, 1)

    stages = [
        FunnelStage(stage="Entry", count=n_entry, drop_off_pct=0.0),
        FunnelStage(stage="Zone Visit", count=n_zone, drop_off_pct=drop_off(n_zone, n_entry)),
        FunnelStage(stage="Billing Queue", count=n_billing, drop_off_pct=drop_off(n_billing, n_zone)),
        FunnelStage(stage="Purchase", count=n_purchase, drop_off_pct=drop_off(n_purchase, n_billing)),
    ]

    return StoreFunnel(
        store_id=store_id,
        date=date_str,
        stages=stages,
        unique_sessions=n_entry,
    )
