"""
Store Intelligence API — FastAPI entrypoint.

Endpoints:
  POST /events/ingest                  — Ingest up to 500 events
  GET  /stores/{store_id}/metrics      — Real-time store metrics
  GET  /stores/{store_id}/funnel       — Conversion funnel
  GET  /stores/{store_id}/heatmap      — Zone heatmap
  GET  /stores/{store_id}/anomalies    — Active anomalies
  GET  /health                         — Service health
  GET  /                               — Dashboard (Web UI)
  GET  /ws/live                        — WebSocket for live metrics
"""

import json
import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Optional

import asyncio
from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import create_tables, get_db, check_db_connection
from app.models import IngestRequest, IngestResult
from app.ingestion import ingest_events
from app.metrics import compute_metrics
from app.funnel import compute_funnel
from app.heatmap import compute_heatmap
from app.anomalies import compute_anomalies
from app.health import compute_health

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    logger.info('{"msg":"Store Intelligence API started"}')
    yield
    logger.info('{"msg":"Store Intelligence API shutting down"}')


app = FastAPI(
    title="Store Intelligence API",
    description="Real-time retail analytics from CCTV + POS data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Structured logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def structured_log_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    start = time.time()

    response = await call_next(request)

    latency_ms = round((time.time() - start) * 1000, 1)
    store_id = request.path_params.get("store_id", "-")

    log_entry = {
        "trace_id": trace_id,
        "store_id": store_id,
        "endpoint": str(request.url.path),
        "method": request.method,
        "latency_ms": latency_ms,
        "status_code": response.status_code,
    }
    logger.info(json.dumps(log_entry))
    return response


# ---------------------------------------------------------------------------
# Exception handlers — no raw stack traces in responses
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    logger.error(f"Unhandled exception trace_id={trace_id}: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again.",
            "trace_id": trace_id,
        },
    )


def _db_unavailable_response():
    return JSONResponse(
        status_code=503,
        content={
            "error": "service_unavailable",
            "message": "Database is temporarily unavailable.",
        },
    )


# ---------------------------------------------------------------------------
# POST /events/ingest
# ---------------------------------------------------------------------------

@app.post("/events/ingest", response_model=IngestResult, status_code=200)
async def ingest(request: Request, payload: IngestRequest, db: Session = Depends(get_db)):
    """
    Ingest up to 500 structured events.
    Idempotent by event_id — duplicate events are silently counted, not re-stored.
    Returns partial success on malformed events.
    """

    try:
        result = ingest_events(payload.events, db)
        # Log with event_count for monitoring
        trace_id = getattr(request.state, "trace_id", "-")
        logger.info(json.dumps({
            "trace_id": trace_id,
            "event_count": len(payload.events),
            "accepted": result.accepted,
            "rejected": result.rejected,
            "duplicate": result.duplicate,
        }))
        return result
    except Exception as exc:
        logger.exception(f"Ingest error: {exc}")
        raise


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/metrics
# ---------------------------------------------------------------------------

@app.get("/stores/{store_id}/metrics")
async def get_metrics(
    store_id: str,
    date_str: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Returns real-time store metrics for today (or a specified date).
    Excludes staff events. Handles zero-purchase stores gracefully.
    """

    target_date = None
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {date_str!r}. Use YYYY-MM-DD.")

    metrics = compute_metrics(store_id, db, target_date)
    return metrics


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/funnel
# ---------------------------------------------------------------------------

@app.get("/stores/{store_id}/funnel")
async def get_funnel(
    store_id: str,
    date_str: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Returns conversion funnel: Entry → Zone Visit → Billing Queue → Purchase.
    Session is the unit — re-entries do not double-count a visitor.
    """

    target_date = None
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {date_str!r}")

    funnel = compute_funnel(store_id, db, target_date)
    return funnel


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/heatmap
# ---------------------------------------------------------------------------

@app.get("/stores/{store_id}/heatmap")
async def get_heatmap(
    store_id: str,
    date_str: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Returns zone visit frequency + avg dwell, normalised 0-100.
    Includes data_confidence=false if fewer than 20 sessions.
    """

    target_date = None
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {date_str!r}")

    heatmap = compute_heatmap(store_id, db, target_date)
    return heatmap


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/anomalies
# ---------------------------------------------------------------------------

@app.get("/stores/{store_id}/anomalies")
async def get_anomalies(
    store_id: str,
    db: Session = Depends(get_db),
):
    """
    Returns active anomalies: queue spike, conversion drop, dead zone, stale feed.
    Each anomaly has a severity (INFO/WARN/CRITICAL) and suggested_action.
    """

    anomalies = compute_anomalies(store_id, db)
    return anomalies


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(db: Session = Depends(get_db)):
    """
    Service health endpoint. Returns last event timestamp per store and STALE_FEED warnings.
    This is what an on-call engineer checks first.
    """
    try:
        from sqlalchemy import text as sql_text
        db.execute(sql_text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    if not db_ok:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "message": "Database unavailable."},
        )
    result = compute_health(db)
    status_code = 200 if result.status == "ok" else 207
    return JSONResponse(content=result.model_dump(), status_code=status_code)


# ---------------------------------------------------------------------------
# WebSocket /ws/live — live metrics stream
# ---------------------------------------------------------------------------

connected_clients: list = []


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # Push metrics every 5 seconds
            try:
                store_id = "STORE_BLR_002"
                metrics = compute_metrics(store_id, db)
                await websocket.send_json({
                    "type": "metrics_update",
                    "store_id": store_id,
                    "data": metrics.model_dump(),
                })
            except Exception as e:
                logger.error(f"WS metrics error: {e}")
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


# ---------------------------------------------------------------------------
# GET / — Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Store Intelligence — Brigade Bangalore</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  header { background: #6d28d9; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.4rem; font-weight: 700; color: #fff; }
  header .badge { background: #10b981; color: #fff; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; padding: 24px; }
  .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
  .card h3 { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
  .card .value { font-size: 2.2rem; font-weight: 700; color: #f8fafc; }
  .card .sub { font-size: 0.8rem; color: #64748b; margin-top: 4px; }
  .card.highlight .value { color: #10b981; }
  .card.warn .value { color: #f59e0b; }
  .card.danger .value { color: #ef4444; }
  .funnel { padding: 0 24px 24px; }
  .funnel h2 { margin-bottom: 12px; font-size: 1rem; color: #94a3b8; }
  .funnel-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  .funnel-bar .label { width: 140px; font-size: 0.85rem; color: #cbd5e1; }
  .funnel-bar .bar-wrap { flex: 1; background: #1e293b; border-radius: 6px; height: 28px; overflow: hidden; }
  .funnel-bar .bar-fill { background: linear-gradient(90deg, #6d28d9, #7c3aed); height: 100%; border-radius: 6px; transition: width 0.6s ease; display: flex; align-items: center; padding-left: 8px; font-size: 0.8rem; color: #e9d5ff; }
  .funnel-bar .drop { width: 70px; font-size: 0.75rem; color: #ef4444; text-align: right; }
  .heatmap-section { padding: 0 24px 24px; }
  .heatmap-section h2 { margin-bottom: 12px; font-size: 1rem; color: #94a3b8; }
  .heatmap-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; }
  .hz { background: #1e293b; border-radius: 8px; padding: 12px; text-align: center; border: 1px solid #334155; }
  .hz .zone-name { font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px; }
  .hz .score-bar { height: 6px; border-radius: 3px; margin: 6px 0; }
  .hz .score { font-size: 0.7rem; color: #64748b; }
  .anomaly-section { padding: 0 24px 24px; }
  .anomaly-section h2 { margin-bottom: 12px; font-size: 1rem; color: #94a3b8; }
  .anomaly { background: #1e293b; border-radius: 8px; padding: 14px; margin-bottom: 8px; border-left: 4px solid #334155; }
  .anomaly.INFO { border-color: #3b82f6; }
  .anomaly.WARN { border-color: #f59e0b; }
  .anomaly.CRITICAL { border-color: #ef4444; }
  .anomaly .a-type { font-size: 0.75rem; font-weight: 700; color: #94a3b8; }
  .anomaly .a-desc { font-size: 0.85rem; color: #cbd5e1; margin: 4px 0; }
  .anomaly .a-action { font-size: 0.78rem; color: #6d28d9; }
  .pulse { animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }
  #status-bar { background: #1e293b; padding: 6px 24px; font-size: 0.75rem; color: #64748b; display: flex; gap: 16px; border-bottom: 1px solid #334155; }
  #status-bar .connected { color: #10b981; }
  #status-bar .disconnected { color: #ef4444; }
</style>
</head>
<body>
<header>
  <h1>🏪 Store Intelligence</h1>
  <span style="color:#c4b5fd;font-size:0.9rem;">Brigade Bangalore</span>
  <span class="badge" id="live-badge">LIVE</span>
</header>
<div id="status-bar">
  <span id="ws-status" class="disconnected">⬤ Connecting...</span>
  <span id="last-update">Last update: —</span>
  <span id="event-count">Events today: —</span>
</div>

<div class="grid" id="metrics-grid">
  <div class="card highlight">
    <h3>Unique Visitors</h3>
    <div class="value" id="m-visitors">—</div>
    <div class="sub">Today</div>
  </div>
  <div class="card">
    <h3>Conversion Rate</h3>
    <div class="value" id="m-conv">—</div>
    <div class="sub">Visitors → Purchase</div>
  </div>
  <div class="card">
    <h3>Avg Basket Value</h3>
    <div class="value" id="m-basket">—</div>
    <div class="sub">INR per transaction</div>
  </div>
  <div class="card warn">
    <h3>Queue Depth Now</h3>
    <div class="value" id="m-queue">—</div>
    <div class="sub">Billing counter</div>
  </div>
  <div class="card">
    <h3>Abandonment Rate</h3>
    <div class="value" id="m-abandon">—</div>
    <div class="sub">Left billing without purchase</div>
  </div>
  <div class="card">
    <h3>Transactions Today</h3>
    <div class="value" id="m-txns">—</div>
    <div class="sub">POS records</div>
  </div>
</div>

<div class="funnel">
  <h2>Conversion Funnel</h2>
  <div id="funnel-bars"></div>
</div>

<div class="heatmap-section">
  <h2>Zone Heatmap</h2>
  <div class="heatmap-grid" id="heatmap-grid"></div>
</div>

<div class="anomaly-section">
  <h2>Active Anomalies</h2>
  <div id="anomaly-list"><p style="color:#64748b;font-size:0.85rem;">Checking...</p></div>
</div>

<script>
const STORE_ID = 'STORE_BLR_002';
const API = '';

function fmt(v, pct) {
  if (v === null || v === undefined) return '—';
  if (pct) return (v * 100).toFixed(1) + '%';
  if (typeof v === 'number') return v.toLocaleString();
  return v;
}

async function fetchMetrics() {
  try {
    const r = await fetch(`${API}/stores/${STORE_ID}/metrics`);
    const d = await r.json();
    document.getElementById('m-visitors').textContent = fmt(d.unique_visitors);
    document.getElementById('m-conv').textContent = fmt(d.conversion_rate, true);
    document.getElementById('m-basket').textContent = '₹' + fmt(Math.round(d.avg_basket_value_inr));
    document.getElementById('m-queue').textContent = fmt(d.queue_depth_now);
    document.getElementById('m-abandon').textContent = fmt(d.abandonment_rate, true);
    document.getElementById('m-txns').textContent = fmt(d.total_transactions);
    document.getElementById('last-update').textContent = 'Last update: ' + new Date().toLocaleTimeString();
  } catch(e) { console.error(e); }
}

async function fetchFunnel() {
  try {
    const r = await fetch(`${API}/stores/${STORE_ID}/funnel`);
    const d = await r.json();
    const container = document.getElementById('funnel-bars');
    container.innerHTML = '';
    const maxCount = d.stages[0]?.count || 1;
    d.stages.forEach(stage => {
      const pct = (stage.count / maxCount * 100).toFixed(1);
      container.innerHTML += `
        <div class="funnel-bar">
          <div class="label">${stage.stage}</div>
          <div class="bar-wrap">
            <div class="bar-fill" style="width:${pct}%">${stage.count}</div>
          </div>
          <div class="drop">${stage.drop_off_pct > 0 ? '-' + stage.drop_off_pct + '%' : ''}</div>
        </div>`;
    });
  } catch(e) { console.error(e); }
}

async function fetchHeatmap() {
  try {
    const r = await fetch(`${API}/stores/${STORE_ID}/heatmap`);
    const d = await r.json();
    const grid = document.getElementById('heatmap-grid');
    grid.innerHTML = '';
    const cells = d.cells.slice(0, 16);
    cells.forEach(cell => {
      const hue = Math.round(260 - cell.normalized_score * 2.2);
      const sat = 60 + cell.normalized_score * 0.3;
      const color = `hsl(${hue},${sat}%,50%)`;
      grid.innerHTML += `
        <div class="hz">
          <div class="zone-name">${cell.zone_name}</div>
          <div class="score-bar" style="background:${color};width:${cell.normalized_score}%"></div>
          <div class="score">${cell.visit_frequency} visits · ${cell.avg_dwell_sec}s</div>
        </div>`;
    });
    if (!d.data_confidence) {
      grid.innerHTML += '<div style="color:#f59e0b;font-size:0.75rem;grid-column:1/-1;padding:8px 0;">⚠ Low data confidence (&lt;20 sessions)</div>';
    }
  } catch(e) { console.error(e); }
}

async function fetchAnomalies() {
  try {
    const r = await fetch(`${API}/stores/${STORE_ID}/anomalies`);
    const d = await r.json();
    const list = document.getElementById('anomaly-list');
    if (!d.active_anomalies || d.active_anomalies.length === 0) {
      list.innerHTML = '<p style="color:#10b981;font-size:0.85rem;">✓ No active anomalies</p>';
      return;
    }
    list.innerHTML = d.active_anomalies.map(a => `
      <div class="anomaly ${a.severity}">
        <div class="a-type">${a.severity} · ${a.anomaly_type.replace(/_/g,' ')}</div>
        <div class="a-desc">${a.description}</div>
        <div class="a-action">→ ${a.suggested_action}</div>
      </div>`).join('');
  } catch(e) { console.error(e); }
}

function startWebSocket() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${protocol}://${location.host}/ws/live`);
  ws.onopen = () => {
    document.getElementById('ws-status').textContent = '⬤ Connected';
    document.getElementById('ws-status').className = 'connected';
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'metrics_update') {
      const d = msg.data;
      document.getElementById('m-visitors').textContent = fmt(d.unique_visitors);
      document.getElementById('m-conv').textContent = fmt(d.conversion_rate, true);
      document.getElementById('last-update').textContent = 'Live · ' + new Date().toLocaleTimeString();
    }
  };
  ws.onclose = () => {
    document.getElementById('ws-status').textContent = '⬤ Disconnected';
    document.getElementById('ws-status').className = 'disconnected';
    setTimeout(startWebSocket, 5000);
  };
}

// Initial load
fetchMetrics();
fetchFunnel();
fetchHeatmap();
fetchAnomalies();

// Poll every 10s
setInterval(() => { fetchMetrics(); fetchAnomalies(); }, 10000);
setInterval(fetchFunnel, 30000);
setInterval(fetchHeatmap, 60000);

startWebSocket();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Live dashboard — shows real-time store metrics as events flow in."""
    return HTMLResponse(content=DASHBOARD_HTML)
