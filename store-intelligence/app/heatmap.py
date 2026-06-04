"""
Zone heatmap computation.

Returns zone visit frequency + avg dwell, normalised 0-100 for grid rendering.
Includes data_confidence flag when fewer than 20 sessions in window.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import StoreHeatmap, HeatmapCell

logger = logging.getLogger(__name__)

MIN_SESSIONS_FOR_CONFIDENCE = 20

# Zone display name + sku_zone mapping (mirrors store_layout.json)
ZONE_META = {
    "ENTRY": ("Entry / Exit", "ENTRY"),
    "BACKLIT": ("Backlit Display", "BACKLIT"),
    "FOH": ("Front of House", "GENERAL"),
    "FRAGRANCE": ("Fragrance", "FRAGRANCE"),
    "NAIL_UNIT": ("Nail Unit", "NAIL"),
    "MAKEUP_UNIT": ("Makeup Unit", "MAKEUP"),
    "EB_KOREAN": ("EB Korean", "SKINCARE"),
    "THE_FACE_SHOP": ("The Face Shop", "SKINCARE"),
    "GOOD_VIBES": ("Good Vibes", "SKINCARE"),
    "DERMDOC": ("DermDoc", "SKINCARE"),
    "MINIMALIST": ("Minimalist", "SKINCARE"),
    "AQUALOGICA": ("Aqualogica", "SKINCARE"),
    "LAKME_SKIN": ("Lakme Skin", "SKINCARE"),
    "ACCESSORIES": ("Accessories", "ACCESSORIES"),
    "CASH_COUNTER": ("Cash Counter", "BILLING"),
    "PMU": ("PMU Station", "SERVICES"),
    "MAYBELLINE": ("Maybelline", "MAKEUP"),
    "FACES_CANADA": ("Faces Canada", "MAKEUP"),
    "LAKME": ("Lakme", "MAKEUP"),
    "COLORBAR_SUGAR": ("Colorbar + Sugar", "MAKEUP"),
    "SWISS_BEAUTY": ("Swiss Beauty", "MAKEUP"),
    "RENEE_NY_BAE": ("Renee NY Bae", "MAKEUP"),
    "ALPS_GOODNESS": ("Alps Goodness", "HAIRCARE"),
    "STREAX": ("Streax", "HAIRCARE"),
}


def compute_heatmap(
    store_id: str, db: Session, target_date: Optional[date] = None
) -> StoreHeatmap:
    if target_date is None:
        target_date = date(2026, 4, 10)

    date_start = datetime.combine(target_date, datetime.min.time()).replace(
        tzinfo=timezone.utc
    )
    date_end = date_start + timedelta(days=1)
    date_str = target_date.isoformat()

    # Zone visit frequency and avg dwell for customer sessions
    rows = db.execute(
        text("""
            SELECT
                zone_id,
                COUNT(*) AS visit_count,
                AVG(dwell_ms) AS avg_dwell_ms
            FROM events
            WHERE store_id = :sid
              AND event_type IN ('ZONE_ENTER', 'ZONE_DWELL', 'ZONE_EXIT')
              AND is_staff = 0
              AND zone_id IS NOT NULL
              AND zone_id NOT IN ('ENTRY', 'EXIT')
              AND timestamp >= :start
              AND timestamp < :end
            GROUP BY zone_id
            ORDER BY visit_count DESC
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchall()

    # Unique session count for confidence flag
    session_rows = db.execute(
        text("""
            SELECT COUNT(DISTINCT visitor_id)
            FROM events
            WHERE store_id = :sid
              AND event_type IN ('ENTRY', 'REENTRY')
              AND is_staff = 0
              AND timestamp >= :start
              AND timestamp < :end
        """),
        {"sid": store_id, "start": date_start, "end": date_end},
    ).fetchone()
    unique_sessions = int(session_rows[0]) if session_rows else 0
    data_confidence = unique_sessions >= MIN_SESSIONS_FOR_CONFIDENCE

    if not rows:
        return StoreHeatmap(
            store_id=store_id,
            date=date_str,
            cells=[],
            data_confidence=data_confidence,
        )

    # Normalize scores 0-100
    max_visits = max(row[1] for row in rows) or 1

    cells = []
    for row in rows:
        zone_id = row[0]
        visit_count = int(row[1])
        avg_dwell_ms = float(row[2] or 0)
        zone_name, sku_zone = ZONE_META.get(zone_id, (zone_id, None))

        normalized = round((visit_count / max_visits) * 100, 1)

        cells.append(
            HeatmapCell(
                zone_id=zone_id,
                zone_name=zone_name,
                visit_frequency=visit_count,
                avg_dwell_sec=round(avg_dwell_ms / 1000.0, 1),
                normalized_score=normalized,
                sku_zone=sku_zone,
            )
        )

    return StoreHeatmap(
        store_id=store_id,
        date=date_str,
        cells=cells,
        data_confidence=data_confidence,
    )
