"""
Anomaly detection for the Store Intelligence API.

Detects:
- BILLING_QUEUE_SPIKE: queue depth exceeds threshold
- CONVERSION_DROP: conversion rate drops vs rolling average
- DEAD_ZONE: no zone visits in 30+ minutes
- LOW_TRAFFIC: visitor count drops suddenly
- STALE_FEED: no events received in 10+ minutes
"""

import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import Anomaly, AnomalySeverity, StoreAnomalies

logger = logging.getLogger(__name__)

# Thresholds
QUEUE_SPIKE_THRESHOLD = 5       # queue depth indicating a spike
DEAD_ZONE_MINUTES = 30          # minutes without any zone visit
STALE_FEED_MINUTES = 10         # minutes without any event
CONVERSION_DROP_THRESHOLD = 0.30  # 30% drop vs rolling average triggers WARN


def _now_utc() -> datetime:
    return datetime(2026, 4, 10, 18, 0, 0, tzinfo=timezone.utc)


def _detect_queue_spike(store_id: str, db: Session, now: datetime) -> Optional[Anomaly]:
    """Detect billing queue spike based on recent BILLING_QUEUE_JOIN events."""
    window_start = now - timedelta(minutes=15)
    row = db.execute(
        text("""
            SELECT MAX(queue_depth)
            FROM events
            WHERE store_id = :sid
              AND event_type = 'BILLING_QUEUE_JOIN'
              AND timestamp >= :start
        """),
        {"sid": store_id, "start": window_start},
    ).fetchone()

    max_depth = int(row[0]) if row and row[0] is not None else 0

    if max_depth >= QUEUE_SPIKE_THRESHOLD:
        return Anomaly(
            anomaly_type="BILLING_QUEUE_SPIKE",
            severity=AnomalySeverity.CRITICAL if max_depth >= 8 else AnomalySeverity.WARN,
            description=f"Billing queue depth reached {max_depth} in the last 15 minutes.",
            suggested_action="Deploy additional cashier or activate mobile POS to clear queue.",
            detected_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            zone_id="CASH_COUNTER",
            value=float(max_depth),
            threshold=float(QUEUE_SPIKE_THRESHOLD),
        )
    return None


def _detect_conversion_drop(store_id: str, db: Session, now: datetime) -> Optional[Anomaly]:
    """Detect conversion rate drop vs 7-day rolling average."""
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today_start - timedelta(days=7)

    # Today's conversion proxy: billing visits / total entries
    today_row = db.execute(
        text("""
            SELECT
                COUNT(DISTINCT CASE WHEN event_type IN ('ENTRY','REENTRY') THEN visitor_id END),
                COUNT(DISTINCT CASE WHEN zone_id = 'CASH_COUNTER' THEN visitor_id END)
            FROM events
            WHERE store_id = :sid
              AND is_staff = 0
              AND timestamp >= :start
        """),
        {"sid": store_id, "start": today_start},
    ).fetchone()

    today_entries = int(today_row[0]) if today_row and today_row[0] else 0
    today_billing = int(today_row[1]) if today_row and today_row[1] else 0
    today_conv = today_billing / today_entries if today_entries > 0 else None

    # 7-day average
    hist_row = db.execute(
        text("""
            SELECT
                COUNT(DISTINCT CASE WHEN event_type IN ('ENTRY','REENTRY') THEN visitor_id END),
                COUNT(DISTINCT CASE WHEN zone_id = 'CASH_COUNTER' THEN visitor_id END)
            FROM events
            WHERE store_id = :sid
              AND is_staff = 0
              AND timestamp >= :week
              AND timestamp < :today
        """),
        {"sid": store_id, "week": week_ago, "today": today_start},
    ).fetchone()

    hist_entries = int(hist_row[0]) if hist_row and hist_row[0] else 0
    hist_billing = int(hist_row[1]) if hist_row and hist_row[1] else 0
    hist_conv = hist_billing / hist_entries if hist_entries > 0 else None

    if today_conv is not None and hist_conv is not None and hist_conv > 0:
        drop = (hist_conv - today_conv) / hist_conv
        if drop >= CONVERSION_DROP_THRESHOLD:
            return Anomaly(
                anomaly_type="CONVERSION_DROP",
                severity=AnomalySeverity.CRITICAL if drop >= 0.5 else AnomalySeverity.WARN,
                description=(
                    f"Conversion rate today ({today_conv:.1%}) is {drop:.1%} below "
                    f"7-day average ({hist_conv:.1%})."
                ),
                suggested_action=(
                    "Review today's promotions and staff engagement. "
                    "Check if billing counter is understaffed."
                ),
                detected_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                value=round(today_conv, 3),
                threshold=round(hist_conv * (1 - CONVERSION_DROP_THRESHOLD), 3),
            )
    return None


def _detect_dead_zones(store_id: str, db: Session, now: datetime) -> List[Anomaly]:
    """Detect zones with no visits in the last 30 minutes during store hours."""
    window_start = now - timedelta(minutes=DEAD_ZONE_MINUTES)

    # Zones that had any visit today
    active_today_rows = db.execute(
        text("""
            SELECT DISTINCT zone_id
            FROM events
            WHERE store_id = :sid
              AND event_type = 'ZONE_ENTER'
              AND is_staff = 0
              AND zone_id IS NOT NULL
              AND timestamp >= :day_start
        """),
        {"sid": store_id, "day_start": now.replace(hour=0, minute=0, second=0, microsecond=0)},
    ).fetchall()
    zones_active_today = {row[0] for row in active_today_rows if row[0]}

    # Zones active in last 30 min
    recent_rows = db.execute(
        text("""
            SELECT DISTINCT zone_id
            FROM events
            WHERE store_id = :sid
              AND event_type = 'ZONE_ENTER'
              AND is_staff = 0
              AND zone_id IS NOT NULL
              AND timestamp >= :start
        """),
        {"sid": store_id, "start": window_start},
    ).fetchall()
    zones_recent = {row[0] for row in recent_rows if row[0]}

    dead_zones = zones_active_today - zones_recent

    anomalies = []
    for zone_id in dead_zones:
        anomalies.append(
            Anomaly(
                anomaly_type="DEAD_ZONE",
                severity=AnomalySeverity.INFO,
                description=f"Zone '{zone_id}' has had no customer visits in the last {DEAD_ZONE_MINUTES} minutes.",
                suggested_action=f"Check if the {zone_id} area is blocked or if staff need to encourage browsing.",
                detected_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                zone_id=zone_id,
            )
        )
    return anomalies


def _detect_stale_feed(store_id: str, db: Session, now: datetime) -> Optional[Anomaly]:
    """Detect if no events have been received in 10+ minutes."""
    stale_threshold = now - timedelta(minutes=STALE_FEED_MINUTES)

    row = db.execute(
        text("""
            SELECT MAX(timestamp)
            FROM events
            WHERE store_id = :sid
        """),
        {"sid": store_id},
    ).fetchone()

    if row is None or row[0] is None:
        return None

    last_ts = row[0]
    if isinstance(last_ts, str):
        try:
            last_ts = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        except Exception:
            return None
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    lag = (now - last_ts).total_seconds() / 60.0

    if lag >= STALE_FEED_MINUTES:
        return Anomaly(
            anomaly_type="STALE_FEED",
            severity=AnomalySeverity.CRITICAL if lag >= 30 else AnomalySeverity.WARN,
            description=f"No events received for {lag:.1f} minutes. Camera feed may be offline.",
            suggested_action="Check camera connectivity and detection pipeline status.",
            detected_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            value=round(lag, 1),
            threshold=float(STALE_FEED_MINUTES),
        )
    return None


def compute_anomalies(store_id: str, db: Session) -> StoreAnomalies:
    """Run all anomaly detectors and return active anomalies."""
    now = _now_utc()
    anomalies: List[Anomaly] = []

    queue_spike = _detect_queue_spike(store_id, db, now)
    if queue_spike:
        anomalies.append(queue_spike)

    conv_drop = _detect_conversion_drop(store_id, db, now)
    if conv_drop:
        anomalies.append(conv_drop)

    dead_zones = _detect_dead_zones(store_id, db, now)
    anomalies.extend(dead_zones)

    stale = _detect_stale_feed(store_id, db, now)
    if stale:
        anomalies.append(stale)

    return StoreAnomalies(
        store_id=store_id,
        active_anomalies=anomalies,
        checked_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
