"""
Event schema definition and emission utilities.
Produces structured JSONL events from detection pipeline output.
"""

import uuid
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

EVENT_TYPES = {
    "ENTRY",
    "EXIT",
    "ZONE_ENTER",
    "ZONE_EXIT",
    "ZONE_DWELL",
    "BILLING_QUEUE_JOIN",
    "BILLING_QUEUE_ABANDON",
    "REENTRY",
}


@dataclass
class EventMetadata:
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = 0


@dataclass
class StoreEvent:
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: str
    timestamp: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = 1.0
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class EventEmitter:
    def __init__(self, output_path: str, store_id: str):
        self.output_path = output_path
        self.store_id = store_id
        self._fh = open(output_path, "a")
        self._count = 0

    def emit(self, event: StoreEvent):
        line = event.to_json()
        self._fh.write(line + "\n")
        self._fh.flush()
        self._count += 1
        logger.debug(f"Emitted {event.event_type} for {event.visitor_id} at {event.timestamp}")

    def close(self):
        self._fh.close()
        logger.info(f"EventEmitter closed — {self._count} events written to {self.output_path}")

    def make_event(
        self,
        camera_id: str,
        visitor_id: str,
        event_type: str,
        timestamp: datetime,
        zone_id: Optional[str] = None,
        dwell_ms: int = 0,
        is_staff: bool = False,
        confidence: float = 1.0,
        queue_depth: Optional[int] = None,
        sku_zone: Optional[str] = None,
        session_seq: int = 0,
    ) -> StoreEvent:
        if isinstance(timestamp, datetime):
            ts_str = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            ts_str = str(timestamp)

        return StoreEvent(
            store_id=self.store_id,
            camera_id=camera_id,
            visitor_id=visitor_id,
            event_type=event_type,
            timestamp=ts_str,
            zone_id=zone_id,
            dwell_ms=dwell_ms,
            is_staff=is_staff,
            confidence=confidence,
            metadata=EventMetadata(
                queue_depth=queue_depth,
                sku_zone=sku_zone,
                session_seq=session_seq,
            ),
        )
