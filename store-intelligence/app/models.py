"""
Pydantic models for the Store Intelligence API.
Covers both ingest event schema and API response schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator
import uuid


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = 0


class StoreEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: EventType
    timestamp: str  # ISO-8601 UTC
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v: str) -> str:
        if not v:
            raise ValueError("event_id must be a non-empty string")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"timestamp must be ISO-8601: {v!r}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return round(v, 4)


class IngestRequest(BaseModel):
    events: List[StoreEvent] = Field(max_length=500)


class IngestResult(BaseModel):
    accepted: int
    rejected: int
    duplicate: int
    errors: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Metrics response models
# ---------------------------------------------------------------------------

class ZoneDwellMetric(BaseModel):
    zone_id: str
    avg_dwell_sec: float
    visit_count: int


class StoreMetrics(BaseModel):
    store_id: str
    date: str
    unique_visitors: int
    conversion_rate: float
    avg_basket_value_inr: float
    avg_dwell_per_zone: List[ZoneDwellMetric]
    queue_depth_now: int
    abandonment_rate: float
    total_transactions: int
    as_of: str  # ISO timestamp of when metrics were computed


# ---------------------------------------------------------------------------
# Funnel response models
# ---------------------------------------------------------------------------

class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float


class StoreFunnel(BaseModel):
    store_id: str
    date: str
    stages: List[FunnelStage]
    unique_sessions: int


# ---------------------------------------------------------------------------
# Heatmap response models
# ---------------------------------------------------------------------------

class HeatmapCell(BaseModel):
    zone_id: str
    zone_name: str
    visit_frequency: int
    avg_dwell_sec: float
    normalized_score: float  # 0–100
    sku_zone: Optional[str] = None


class StoreHeatmap(BaseModel):
    store_id: str
    date: str
    cells: List[HeatmapCell]
    data_confidence: bool  # False if fewer than 20 sessions


# ---------------------------------------------------------------------------
# Anomaly response models
# ---------------------------------------------------------------------------

class AnomalySeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class Anomaly(BaseModel):
    anomaly_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    anomaly_type: str
    severity: AnomalySeverity
    description: str
    suggested_action: str
    detected_at: str
    zone_id: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


class StoreAnomalies(BaseModel):
    store_id: str
    active_anomalies: List[Anomaly]
    checked_at: str


# ---------------------------------------------------------------------------
# Health response models
# ---------------------------------------------------------------------------

class StoreHealth(BaseModel):
    store_id: str
    last_event_ts: Optional[str]
    is_stale: bool
    lag_minutes: Optional[float] = None
    event_count_today: int


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    stores: List[StoreHealth]
    db_connected: bool
    checked_at: str
