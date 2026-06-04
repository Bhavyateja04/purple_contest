# Store Intelligence — Brigade Bangalore

> End-to-end retail analytics pipeline: raw CCTV → live store metrics.

## Quick Start (5 Commands)

```bash
# 1. Clone and enter the project
cd store-intelligence

# 2. Create the empty events file (needed for Docker volume mount)
touch events.jsonl

# 3. Start the API
docker compose up --build -d

# 4. Verify the API is running
curl http://localhost:8000/health

# 5. Open the live dashboard
open http://localhost:8000
```

The API is live at **http://localhost:8000**. The interactive dashboard is at the root URL.

---

## Running the Detection Pipeline

### Prerequisites

```bash
# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (includes YOLOv8, OpenCV, ByteTrack)
pip install -r requirements.txt

# YOLOv8 will auto-download yolov8n.pt on first run (~6MB)
```

### Process All Camera Clips

```bash
bash pipeline/run.sh
```

This processes all 5 camera clips against the store layout, emitting structured events to `events.jsonl`.

### Process a Single Camera

```bash
python pipeline/detect.py \
    --clip "../CCTV Footage/CAM 1.mp4" \
    --camera CAM_ENTRY_01 \
    --output events.jsonl \
    --pos "../Brigade_Bangalore_10_April_26 (1)bc6219c.csv"
```

### Feed Events into the API

```bash
# Feed events.jsonl into the running API in batches of 500
python3 -c "
import json, requests
with open('events.jsonl') as f:
    events = [json.loads(l) for l in f if l.strip()]

BATCH = 500
for i in range(0, len(events), BATCH):
    batch = events[i:i+BATCH]
    r = requests.post('http://localhost:8000/events/ingest', json={'events': batch})
    result = r.json()
    print(f'Batch {i//BATCH+1}: accepted={result[\"accepted\"]}, dup={result[\"duplicate\"]}')

print('Done.')
"
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/events/ingest` | Ingest up to 500 events. Idempotent by `event_id`. |
| `GET`  | `/stores/{id}/metrics` | Today's unique visitors, conversion rate, queue depth, etc. |
| `GET`  | `/stores/{id}/funnel` | Entry → Zone Visit → Billing → Purchase funnel |
| `GET`  | `/stores/{id}/heatmap` | Zone frequency heatmap, normalised 0–100 |
| `GET`  | `/stores/{id}/anomalies` | Active anomalies with severity and suggested actions |
| `GET`  | `/health` | Service health + per-store staleness |
| `GET`  | `/` | Live dashboard (Web UI) |
| `WS`   | `/ws/live` | WebSocket stream for real-time metric updates |

### Example Requests

```bash
# Store metrics
curl http://localhost:8000/stores/STORE_BLR_002/metrics

# Conversion funnel for a specific date
curl "http://localhost:8000/stores/STORE_BLR_002/funnel?date_str=2026-04-10"

# Zone heatmap
curl http://localhost:8000/stores/STORE_BLR_002/heatmap

# Active anomalies
curl http://localhost:8000/stores/STORE_BLR_002/anomalies
```

---

## Terminal Live Dashboard (Part E)

```bash
# With API running and events loaded:
pip install rich requests
python dashboard/live_dashboard.py --api http://localhost:8000 --store STORE_BLR_002
```

Shows metrics, funnel, heatmap, and anomalies updating every 5 seconds.

---

## Running Tests

```bash
# From the store-intelligence/ directory
pip install -r requirements.txt

# Run all tests with coverage
pytest tests/ -v --cov=app --cov=pipeline --cov-report=term-missing

# Run specific test files
pytest tests/test_pipeline.py -v
pytest tests/test_metrics.py -v
pytest tests/test_anomalies.py -v
```

---

## Project Structure

```
store-intelligence/
├── pipeline/
│   ├── detect.py          # Main detection + tracking (YOLOv8 + ByteTrack)
│   ├── tracker.py         # Re-ID / tracking state machine
│   ├── emit.py            # Event schema + JSONL emission
│   └── run.sh             # One-command clip processor
├── app/
│   ├── main.py            # FastAPI entrypoint + dashboard
│   ├── models.py          # Pydantic event + response schemas
│   ├── database.py        # SQLAlchemy + SQLite setup
│   ├── ingestion.py       # Ingest, dedup, partial success
│   ├── metrics.py         # Real-time metric computation + POS correlation
│   ├── funnel.py          # Session-level conversion funnel
│   ├── heatmap.py         # Zone heatmap (0–100 normalised)
│   ├── anomalies.py       # Anomaly detectors
│   └── health.py          # Health check + STALE_FEED detection
├── dashboard/
│   └── live_dashboard.py  # Terminal live dashboard (rich)
├── tests/
│   ├── test_pipeline.py   # Detection pipeline + tracker tests
│   ├── test_metrics.py    # API endpoint tests (TestClient + in-memory SQLite)
│   └── test_anomalies.py  # Anomaly detector unit tests
├── docs/
│   ├── DESIGN.md          # Architecture + AI-assisted decisions
│   └── CHOICES.md         # 3 key decisions with reasoning
├── store_layout.json      # Zone definitions for Brigade Bangalore
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DATABASE_URL` | `sqlite:////data/store_intelligence.db` | SQLAlchemy DB URL |
| `POS_CSV_PATH` | Auto-detected from parent dir | Path to POS transactions CSV |

---

## Architecture Summary

1. **Detection**: YOLOv8-nano detects persons, ByteTrack maintains stable tracker IDs across frames, `ReIDTracker` assigns `visitor_id`s and handles re-entry detection via color histogram similarity.
2. **Events**: Structured JSONL with 8 event types, each with full provenance (camera, zone, confidence, staff flag).
3. **API**: FastAPI with SQLite storage. All endpoints are real-time (no caching). Idempotent ingest.
4. **Dashboard**: Single-page web app at `/` + WebSocket at `/ws/live` for live updates.

See `docs/DESIGN.md` for the full architecture and `docs/CHOICES.md` for decision rationale.
