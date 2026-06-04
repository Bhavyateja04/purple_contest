"""
Event ingestion logic.
Handles validation, deduplication (idempotent by event_id), and storage.
"""

import logging
from datetime import datetime, timezone
from typing import List, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models import StoreEvent, IngestResult
from app.database import EventRecord

logger = logging.getLogger(__name__)


def ingest_events(
    events: List[StoreEvent], db: Session
) -> IngestResult:
    """
    Ingest a batch of events.
    - Idempotent: events with duplicate event_id are silently skipped.
    - Partial success: malformed events are rejected with error details.
    - Returns counts of accepted / rejected / duplicate events.
    """
    accepted = 0
    rejected = 0
    duplicate = 0
    errors = []

    # Batch dedup: collect all event_ids present in DB for this batch
    incoming_ids = [e.event_id for e in events]
    existing_ids: set = set()

    if incoming_ids:
        rows = db.execute(
            select(EventRecord.event_id).where(
                EventRecord.event_id.in_(incoming_ids)
            )
        ).fetchall()
        existing_ids = {row[0] for row in rows}

    now_utc = datetime.now(timezone.utc)
    seen_ids: set = set()  # guard against duplicates within same batch

    for event in events:
        try:
            if event.event_id in existing_ids or event.event_id in seen_ids:
                duplicate += 1
                continue

            seen_ids.add(event.event_id)

            # Parse timestamp
            ts_str = event.timestamp.replace("Z", "+00:00")
            event_dt = datetime.fromisoformat(ts_str)

            record = EventRecord(
                event_id=event.event_id,
                store_id=event.store_id,
                camera_id=event.camera_id,
                visitor_id=event.visitor_id,
                event_type=event.event_type.value,
                timestamp=event_dt,
                zone_id=event.zone_id,
                dwell_ms=event.dwell_ms,
                is_staff=event.is_staff,
                confidence=event.confidence,
                queue_depth=event.metadata.queue_depth,
                sku_zone=event.metadata.sku_zone,
                session_seq=event.metadata.session_seq,
                ingested_at=now_utc,
            )
            db.add(record)
            accepted += 1

        except Exception as exc:
            rejected += 1
            errors.append(
                {
                    "event_id": getattr(event, "event_id", "unknown"),
                    "error": str(exc),
                }
            )
            logger.warning(f"Rejected event {getattr(event, 'event_id', '?')}: {exc}")

    if accepted > 0:
        db.commit()
        logger.info(
            f"Ingested batch: accepted={accepted}, rejected={rejected}, duplicate={duplicate}"
        )

    return IngestResult(
        accepted=accepted,
        rejected=rejected,
        duplicate=duplicate,
        errors=errors,
    )
