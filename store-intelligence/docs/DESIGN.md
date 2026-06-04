# DESIGN.md — Store Intelligence System

## Architecture Overview

The system is a four-stage pipeline that converts raw CCTV footage into live business analytics for Apex Retail's Brigade Bangalore store.

```
┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  CCTV Clips  │───▶│ Detection Layer  │───▶│  Event Stream     │───▶│ Intelligence API │
│  (5 cameras) │    │ YOLOv8 + ByteTrack│   │  events.jsonl     │    │ FastAPI + SQLite  │
└──────────────┘    └──────────────────┘    └───────────────────┘    └──────────────────┘
                           │                                                    │
                    Re-ID Tracker                                      Live Dashboard
                    Staff Detector                                     WebSocket Feed
                    Zone Assigner                                      Anomaly Engine
```

### Stage 1: Detection Layer (`pipeline/`)

**Entry Point:** `pipeline/detect.py`

The detection layer runs frame-by-frame on each of the 5 camera feeds:

1. **Frame extraction**: OpenCV reads every 3rd frame (5 fps effective at 15fps source) to balance accuracy against processing time.
2. **Person detection**: YOLOv8-nano runs inference on each sampled frame, detecting bounding boxes for class 0 (person) at confidence ≥ 0.25. Low-confidence detections are emitted (not dropped) with their actual confidence score.
3. **Multi-object tracking**: ByteTrack (embedded in ultralytics) assigns stable integer tracker IDs across frames using IoU-based matching with a Kalman filter for motion prediction.
4. **Zone assignment**: Each camera has a dedicated zone-assignment function that maps the bounding box centroid (x, y) to a named zone. Entry camera uses vertical position for ENTRY/EXIT classification; floor and display cameras use horizontal position to segment named brand zones.
5. **Re-ID**: When a new tracker ID appears, a color histogram (HSV H+S channels, 26 bins) of the bounding box crop is compared against recently-exited visitors using cosine similarity. If similarity > 0.80 within a 15-minute window, the detection is flagged as REENTRY.
6. **Staff classification**: A heuristic evaluates each tracker's cumulative zone history. Trackers with ≥ 5 zone transitions, average dwell < 45 seconds per zone, and total presence > 5 minutes are classified as staff (`is_staff=true`). This is a per-tracker rolling evaluation updated every 30 frames.
7. **Event emission**: Structured JSON events are written to `events.jsonl` via the `EventEmitter` class, which handles UTC timestamp conversion and metadata population.

**Entry/Exit Detection**: The entry camera (CAM_ENTRY_01) tracks centroid Y-position over a sliding window of 10 frames. A normalized delta > 0.10 between the first and second halves of the window triggers an ENTRY or EXIT classification. This avoids false triggers from momentary camera vibration or partial occlusions.

**Group Handling**: ByteTrack natively assigns separate tracker IDs to each detected person, so 3 people entering simultaneously produce 3 ENTRY events with 3 distinct `visitor_id` values. No additional grouping logic is needed.

### Stage 2: Event Schema

Events are emitted as JSONL with the schema defined in `pipeline/emit.py` and validated by `app/models.py`. Key design decisions:

- `event_id`: UUID v4, generated at emission time. Globally unique across all cameras and clips.
- `visitor_id`: Hash of `camera_id + tracker_id + timestamp_ns`, prefixed `VIS_`. Stable within a session; reassigned across sessions unless Re-ID matches.
- `timestamp`: Derived from `clip_start_time + (frame_index / fps)`. Stored as ISO-8601 UTC.
- `is_staff`: Boolean flag set by heuristic classifier. Propagates through all events for the tracker.
- `confidence`: Raw YOLO detection confidence, rounded to 4 decimal places. Never suppressed.

### Stage 3: Intelligence API (`app/`)

**Framework**: FastAPI 0.115 with Pydantic v2 for validation.

**Storage**: SQLite via SQLAlchemy with indexed columns on `(store_id, timestamp)`, `(store_id, event_type)`, and `(visitor_id, event_type)` for query performance.

**Endpoints**:

| Endpoint | Logic Summary |
|----------|---------------|
| `POST /events/ingest` | Batch dedup via IN query on `event_id`. Partial success: good events are committed even if some fail validation. Idempotent. |
| `GET /stores/{id}/metrics` | Aggregates from DB with `GROUP BY zone_id`. POS correlation via `_count_converted_visitors()` with 5-minute window join. |
| `GET /stores/{id}/funnel` | Session-level counting using `DISTINCT visitor_id` across four funnel stages. REENTRY events are included with ENTRY in stage 1 but deduplicated by `visitor_id`. |
| `GET /stores/{id}/heatmap` | Zone visit frequency normalized 0-100 relative to max zone. Confidence flag if `DISTINCT visitor_id` count < 20. |
| `GET /stores/{id}/anomalies` | Four independent detectors, each returning an `Anomaly` or `None`. Stale feed checks `MAX(timestamp)` against current time. |
| `GET /health` | Per-store last event timestamp + `is_stale` flag. Returns HTTP 207 if any store is stale. |

**Structured Logging**: Every request logs `trace_id`, `store_id`, `endpoint`, `latency_ms`, `status_code` as JSON. Ingest additionally logs `event_count`, `accepted`, `rejected`, `duplicate`.

**Error handling**: A global exception handler catches all unhandled exceptions and returns a structured JSON error with `trace_id` — no raw stack traces in responses.

### Stage 4: Live Dashboard

The dashboard is a single-page React-equivalent app (vanilla JS) served at `GET /` and backed by a WebSocket at `/ws/live`. Metrics update every 5 seconds via WebSocket push. The funnel uses CSS-animated progress bars. The heatmap renders all zones with color mapped from purple (low) to blue (high) based on normalized score.

---

## AI-Assisted Decisions

### 1. Zone assignment strategy for overlapping camera views

I described the store layout image (Brigade Road floor plan) to Claude and asked: *"Given a store with 5 cameras covering entry, main floor, back display wall, billing counter, and front display wall, how should I assign detected bounding boxes to named zones without a calibration step?"*

Claude suggested using relative bounding box centroid position (cx/frame_width, cy/frame_height) as a 2D coordinate mapped to zones defined as rectangular regions. It also suggested a calibration step using a reference frame.

**What I chose**: I simplified to 1D mapping (x-position for multi-zone cameras, y-position for the entry camera) because the floor plan shows zones arranged in horizontal rows with minimal vertical overlap. This is less general but requires zero calibration and is trivially debuggable. I noted Claude's 2D approach as a future improvement for stores with L-shaped layouts.

### 2. Staff detection heuristic

I asked Claude: *"How would you detect staff in a retail CCTV feed without a separate staff model, relying only on tracking data?"*

Claude suggested three approaches:
- (a) Appearance clustering (staff uniforms appear as a cluster in color space)
- (b) Trajectory analysis (staff cross many zones quickly)
- (c) Re-entry counting (staff appear many times per day)

I implemented **approach (b)** as the primary classifier because it requires no training data and works regardless of uniform color. I store cumulative zone history per tracker and evaluate the heuristic every 30 frames. Claude's suggestion to combine (b) and (c) is noted as an improvement — staff who appear many times (c) and move quickly (b) have higher confidence of being staff.

**Override**: Claude initially suggested a 2-zone threshold for staff detection. I raised it to 5 zones and added a minimum total presence time of 5 minutes. With only 2 zones the false positive rate on quick-browsing customers was too high.

### 3. POS correlation approach for conversion rate

I asked Claude: *"How do I correlate anonymized POS transactions with CCTV visitor sessions when there is no customer ID in either dataset?"*

Claude suggested:
- Time-window correlation: any visitor in the billing zone within N minutes before a transaction counts as "converted"
- Multiple transaction matching: a transaction can only be claimed by one visitor (greedy assignment)

**What I chose**: I implemented the time-window approach with a 5-minute window (matching the problem statement specification). I did not implement greedy assignment because it requires a more complex join and the dataset is small enough that overlap is unlikely. This means conversion rate is slightly inflated if two customers are in the billing zone simultaneously, which I documented as a known limitation.

---

## Known Limitations and Future Work

1. **Cross-camera deduplication**: Currently each camera has its own `ReIDTracker` instance. A customer visible in both the entry camera and the main floor camera will generate events from both — a global Re-ID module using appearance features across cameras would fix this.

2. **Staff detection accuracy**: The heuristic fails for staff who stand at one counter for long periods (e.g., a cashier who never moves). A future improvement would add a position-based rule: anyone always within the `CASH_COUNTER` bounding box is likely staff.

3. **Entry/exit direction**: The entry camera direction heuristic assumes the camera is mounted perpendicular to the door. If the camera is mounted at an angle, the Y-position heuristic may misclassify sideways movement as directional crossing.

4. **Zone assignment calibration**: The current approach uses a fixed 1D linear mapping. A homography-based mapping from pixel coordinates to the store floor plan would be significantly more accurate, particularly for the floor camera where zones overlap near the boundaries.
