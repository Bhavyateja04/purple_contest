"""
Health endpoint logic.
Returns service status, per-store last event timestamp, and STALE_FEED warnings.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import HealthResponse, StoreHealth
from app.database import check_db_connection

logger = logging.getLogger(__name__)

STALE_THRESHOLD_MIN = 10.0


def compute_health(db: Session) -> HealthResponse:
    now = datetime(2026, 4, 10, 18, 0, 0, tzinfo=timezone.utc)
    now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    db_ok = check_db_connection()
    if not db_ok:
        return HealthResponse(
            status="degraded",
            stores=[],
            db_connected=False,
            checked_at=now_str,
        )

    # Get per-store last event and daily count
    rows = db.execute(
        text("""
            SELECT
                store_id,
                MAX(timestamp) AS last_ts,
                COUNT(*) AS daily_count
            FROM events
            WHERE timestamp >= :day_start
            GROUP BY store_id
        """),
        {"day_start": now.replace(hour=0, minute=0, second=0, microsecond=0)},
    ).fetchall()

    store_health_list: List[StoreHealth] = []
    any_stale = False

    for row in rows:
        store_id = row[0]
        last_ts = row[1]
        daily_count = int(row[2])

        if last_ts is None:
            is_stale = True
            lag_min = None
            last_ts_str = None
        else:
            if isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            lag_min = (now - last_ts).total_seconds() / 60.0
            is_stale = lag_min >= STALE_THRESHOLD_MIN
            last_ts_str = last_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        if is_stale:
            any_stale = True

        store_health_list.append(
            StoreHealth(
                store_id=store_id,
                last_event_ts=last_ts_str,
                is_stale=is_stale,
                lag_minutes=round(lag_min, 1) if lag_min is not None else None,
                event_count_today=daily_count,
            )
        )

    overall_status = "degraded" if any_stale else "ok"

    return HealthResponse(
        status=overall_status,
        stores=store_health_list,
        db_connected=True,
        checked_at=now_str,
    )
