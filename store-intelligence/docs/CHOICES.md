# CHOICES.md — Engineering Decisions

## Decision 1: Detection Model — YOLOv8-nano with ByteTrack

### Options Considered

| Model | Pros | Cons |
|-------|------|------|
| **YOLOv8n** (chosen) | Fast on CPU, good person detection, native ByteTrack integration, easy to upgrade to `yolov8s` for accuracy | Slightly lower recall on occluded persons than larger models |
| YOLOv9 | Higher accuracy on benchmarks | Slower, more complex setup, no built-in tracker in ultralytics at time of writing |
| RT-DETR | Transformer-based, strong on crowded scenes | Heavy (requires GPU for real-time), complex dependency chain |
| MediaPipe Pose | Works on-device, no GPU | Only detects pose landmarks, not bounding boxes — incompatible with ByteTrack |
| GPT-4V / Claude Vision (VLM) | Excellent scene understanding | 10–50 API calls per second needed — cost and latency prohibitive for 20-minute clips at 15fps |

### What AI Suggested

I described the use case to Claude and asked for model recommendations. It suggested:
- **YOLOv8s** (small) for accuracy/speed balance at 1080p 15fps
- **DeepSORT** or **StrongSORT** for tracking (both have re-identification modules built-in)
- Using a VLM (GPT-4V) for **zone classification** rather than bounding box position (the idea being that GPT-4V could identify which zone a person is standing in by looking at the frame context)

### What I Chose and Why

**YOLOv8-nano + ByteTrack** (not YOLOv8s as suggested).

Reasoning:
- The nano model runs at ~8fps on a modern CPU for 1080p input — acceptable for our 5fps effective processing rate (FRAME_SKIP=3)
- ByteTrack is the default tracker in ultralytics and requires zero additional configuration
- The model selection is designed to be easily upgraded: swapping `yolov8n.pt` → `yolov8s.pt` is a 1-line change

**On VLM zone classification**: I considered this for 2 days. The appeal is obvious — GPT-4V would be much more accurate at saying "this person is standing in front of the DermDoc display" than my 1D centroid mapping. The problem is latency and cost: at 5 processed frames per second, 5 cameras, and 20 minutes per clip, that's 30,000 API calls per clip. At $0.01 per call, that's $300 per clip. I chose rule-based zone assignment and documented this trade-off. A viable middle ground would be to use a lightweight CLIP-based zone classifier trained on a handful of labeled frames from each camera.

**On StrongSORT**: ByteTrack does not use appearance features for tracking (only IoU + motion), which means it can lose a track when someone steps behind a display. StrongSORT uses both appearance and motion, which would improve tracking accuracy in the occluded billing scene. However, StrongSORT requires a re-identification model (typically OSNet) which adds 200MB to the Docker image and significant latency. I chose ByteTrack for simplicity and implemented a lightweight appearance-based Re-ID layer (`tracker.py`) on top for session-level re-entry detection.

---

## Decision 2: Event Schema Design

### Options Considered

**Option A: Flat schema per event type** — separate schemas for ENTRY events, ZONE_ENTER events, etc.
- Pro: Type-safe, no null fields
- Con: 8 different schemas to maintain, complex Pydantic union types, API ingest logic becomes a type dispatcher

**Option B: Single polymorphic schema with nullable fields** (chosen) — one schema, fields are null when not applicable
- Pro: Simple ingest pipeline (`List[StoreEvent]`), one Pydantic model, easy to add new event types
- Con: Some fields are always null (e.g., `zone_id` for ENTRY events), loose typing

**Option C: Event envelope + typed payload** — outer envelope with `event_type` + `payload: Any`
- Pro: Very extensible
- Con: Kills type safety at ingest, requires dynamic dispatch, complex validation

### What AI Suggested

Claude suggested **Option C** (envelope + typed payload) as "the most production-ready pattern, similar to what Kafka event schemas use." It pointed to the CloudEvents spec as a reference.

### What I Chose and Why

**Option B** (single polymorphic schema), disagreeing with Claude's suggestion.

The envelope pattern is correct for a large event bus where event producers and consumers are decoupled teams with evolving schemas. For this system, the producer (detection pipeline) and consumer (API ingest) are the same codebase and evolve together. The overhead of a dispatch layer and dynamic payload validation is not justified.

Key schema decisions:
- `event_id` as UUID v4 (not a hash of content) so that two identical-looking events from two cameras are always distinct
- `visitor_id` with `VIS_` prefix makes it human-readable in logs and easy to filter
- `is_staff` as a top-level boolean (not nested in metadata) because it changes query structure — `WHERE is_staff = 0` appears in every customer-facing query
- `confidence` kept in the root (not metadata) for the same reason — it's a quality signal that affects all queries, not just zone-specific ones
- `metadata.session_seq` as an ordinal counter per visitor session, useful for debugging event ordering without sorting by timestamp

**What I changed after AI feedback**: Claude's initial schema placed `queue_depth` as a top-level field. I moved it to `metadata` because it is only meaningful for `BILLING_QUEUE_JOIN` events and keeping it top-level polluted every other event type with a null field in a critical position.

---

## Decision 3: API Storage Engine — SQLite over PostgreSQL

### Options Considered

| Engine | Pros | Cons |
|--------|------|------|
| **SQLite** (chosen) | Zero ops, no container, no migrations, works in `docker compose up` without side containers | Single-writer bottleneck, not suitable for >1 API worker, file-based |
| PostgreSQL | Production-grade, multi-writer, mature tooling | Requires additional docker service, connection pooling, migrations setup |
| DuckDB | Excellent analytical query performance | Write throughput lower than SQLite for OLTP, less FastAPI ecosystem support |
| Redis (as primary store) | Sub-millisecond reads | Not a relational DB, complex to query for funnel/heatmap aggregations |

### What AI Suggested

I asked Claude: *"What storage engine should I use for a real-time retail analytics API where events are ingested continuously and queried for aggregated metrics?"*

Claude strongly recommended **PostgreSQL with TimescaleDB** for time-series analytics, citing:
- Hypertables for efficient time-range queries
- Automatic chunk pruning for old data
- `time_bucket()` for window aggregations

It also suggested adding **Redis** as a caching layer between the DB and API.

### What I Chose and Why

**SQLite**, explicitly disagreeing with Claude's recommendation for this context.

Reasoning:
1. **Acceptance gate**: `docker compose up` must start everything without manual steps. PostgreSQL requires either a cloud instance or a second container — adding ~30 seconds to startup and ~100MB to the image. SQLite starts in milliseconds with zero configuration.
2. **Dataset scale**: The challenge dataset is one store, one day, 5 cameras, 20 minutes per clip. Estimated events: 5 cameras × 20 min × 60s × 5fps × 0.3 (average people per frame) × 2 (events per detection) ≈ 18,000 events. SQLite handles this volume trivially.
3. **Query pattern**: All aggregation queries are `GROUP BY zone_id` or `COUNT(DISTINCT visitor_id)` over a single day's worth of data. With the `(store_id, timestamp)` composite index, these run in <10ms on SQLite with 100,000 rows.
4. **Upgradeability**: The storage engine is behind a `DATABASE_URL` environment variable. Switching to PostgreSQL requires: (1) adding postgres service to docker-compose.yml, (2) changing `DATABASE_URL`. The SQLAlchemy layer ensures zero application code changes.

**Where I'd use PostgreSQL instead**: At 40 stores live with continuous event ingestion from 3 cameras each, ingest volume is ~40x higher. At that scale, SQLite's single-writer lock would cause ingest latency to spike during analytics query bursts. That's the correct point to migrate to PostgreSQL (or a purpose-built time-series store).

**On Redis caching**: The metrics endpoints currently run SQL queries on every request. For this dataset size, the latency is <50ms, which doesn't justify the operational complexity of a Redis sidecar. If response latency SLAs tightened below 10ms, I would add a simple TTL cache (5 seconds) using Python's `functools.lru_cache` or `fastapi-cache2`, not Redis.
