"""
Main detection and tracking pipeline.

Processes each CCTV clip with:
1. YOLOv8 person detection
2. ByteTrack multi-object tracking (built into ultralytics)
3. Zone assignment based on camera type and bounding box position
4. Entry/Exit direction detection for entry camera
5. Staff exclusion heuristics
6. Re-ID for re-entry detection
7. Structured event emission

Usage:
    python detect.py --clip "CAM 1.mp4" --camera CAM_ENTRY_01 --output events.jsonl
    python detect.py --all  # process all cameras
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.emit import EventEmitter, StoreEvent
from pipeline.tracker import ReIDTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("detect")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STORE_ID = "STORE_BLR_002"
LAYOUT_FILE = Path(__file__).parent.parent / "store_layout.json"

# Process every Nth frame to balance speed vs accuracy
# At 15fps, processing every 5th frame → 3fps effective (good enough for retail foot traffic)
FRAME_SKIP = 5

# Minimum detection confidence to track (low-conf still emitted with flag)
MIN_CONF = 0.25

# Entry line for entry camera: fraction of frame height
# Objects moving from bottom to top are ENTERING; top to bottom are EXITING
ENTRY_LINE_Y_RATIO = 0.5

# Smoothing window for direction classification (frames)
DIRECTION_WINDOW = 10

# Billing camera: if more than N people in frame simultaneously, it's a queue
QUEUE_DEPTH_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Zone assignment helpers
# ---------------------------------------------------------------------------

def assign_zone_entry_camera(
    cx: float, cy: float, frame_h: float, frame_w: float
) -> str:
    """For the entry camera, assign zone based on vertical position."""
    if cy < frame_h * 0.4:
        return "BACKLIT"
    return "ENTRY"


def assign_zone_floor_camera(
    cx: float, cy: float, frame_h: float, frame_w: float
) -> str:
    """For the main floor camera, assign zone based on x-position."""
    rel_x = cx / frame_w
    if rel_x < 0.25:
        return "FRAGRANCE"
    elif rel_x < 0.45:
        return "NAIL_UNIT"
    elif rel_x < 0.70:
        return "FOH"
    else:
        return "MAKEUP_UNIT"


def assign_zone_display_camera(
    cx: float, cy: float, frame_h: float, frame_w: float
) -> str:
    """For the back display wall camera, assign zone based on x-position."""
    rel_x = cx / frame_w
    zones = [
        "EB_KOREAN", "THE_FACE_SHOP", "GOOD_VIBES", "DERMDOC",
        "MINIMALIST", "AQUALOGICA", "LAKME_SKIN", "ACCESSORIES"
    ]
    idx = min(int(rel_x * len(zones)), len(zones) - 1)
    return zones[idx]


def assign_zone_billing_camera(
    cx: float, cy: float, frame_h: float, frame_w: float
) -> str:
    """For the billing camera, classify as counter or PMU."""
    rel_x = cx / frame_w
    return "CASH_COUNTER" if rel_x < 0.7 else "PMU"


def assign_zone_front_camera(
    cx: float, cy: float, frame_h: float, frame_w: float
) -> str:
    """For the front display camera, assign zone based on x-position."""
    rel_x = cx / frame_w
    zones = [
        "MAYBELLINE", "FACES_CANADA", "LAKME", "COLORBAR_SUGAR",
        "SWISS_BEAUTY", "RENEE_NY_BAE", "ALPS_GOODNESS", "STREAX"
    ]
    idx = min(int(rel_x * len(zones)), len(zones) - 1)
    return zones[idx]


ZONE_ASSIGNERS = {
    "CAM_ENTRY_01": assign_zone_entry_camera,
    "CAM_FLOOR_02": assign_zone_floor_camera,
    "CAM_DISPLAY_03": assign_zone_display_camera,
    "CAM_BILLING_04": assign_zone_billing_camera,
    "CAM_FRONT_05": assign_zone_front_camera,
}

# Which camera handles entry/exit detection
ENTRY_CAMERAS = {"CAM_ENTRY_01"}
BILLING_CAMERAS = {"CAM_BILLING_04"}


# ---------------------------------------------------------------------------
# Direction tracker for entry camera
# ---------------------------------------------------------------------------

class DirectionTracker:
    """
    Tracks centroid Y-position over time to determine entry vs exit direction.
    Entry: centroid moves from high Y (bottom of frame) to low Y (top) — inward.
    Exit: centroid moves from low Y to high Y.
    """

    def __init__(self, window: int = DIRECTION_WINDOW):
        self.window = window
        self._positions: Dict[int, List[float]] = {}
        self._crossed: Dict[int, str] = {}  # tracker_id → 'ENTRY' | 'EXIT' | None

    def update(
        self, tracker_id: int, cy_norm: float, frame_h: float
    ) -> Optional[str]:
        """
        Update position history. Returns 'ENTRY', 'EXIT', or None if no crossing yet.
        cy_norm: normalized y coordinate (0=top, 1=bottom)
        """
        if tracker_id not in self._positions:
            self._positions[tracker_id] = []
        self._positions[tracker_id].append(cy_norm)
        # Keep only last N positions
        self._positions[tracker_id] = self._positions[tracker_id][-self.window:]

        if tracker_id in self._crossed:
            return None  # Already classified this tracker

        positions = self._positions[tracker_id]
        if len(positions) < 4:
            return None

        first_avg = np.mean(positions[: len(positions) // 2])
        second_avg = np.mean(positions[len(positions) // 2 :])
        delta = second_avg - first_avg

        # Significant downward movement (toward top = entry) — depends on camera orientation
        # For entry camera facing door: people entering move away from door = upward in frame
        # This depends on how the camera is mounted. We use a heuristic:
        # if centroid started near bottom-center and moved toward top → ENTRY
        if abs(delta) > 0.10:
            direction = "EXIT" if delta > 0 else "ENTRY"
            self._crossed[tracker_id] = direction
            return direction
        return None

    def remove(self, tracker_id: int):
        self._positions.pop(tracker_id, None)
        self._crossed.pop(tracker_id, None)


# ---------------------------------------------------------------------------
# Main clip processor
# ---------------------------------------------------------------------------

class ClipProcessor:
    def __init__(
        self,
        clip_path: str,
        camera_id: str,
        emitter: EventEmitter,
        clip_start_time: datetime,
        layout: dict,
    ):
        self.clip_path = clip_path
        self.camera_id = camera_id
        self.emitter = emitter
        self.clip_start_time = clip_start_time
        self.layout = layout
        self.reid_tracker = ReIDTracker(camera_id)
        self.dir_tracker = DirectionTracker()
        self._zone_assigner = ZONE_ASSIGNERS.get(camera_id, assign_zone_floor_camera)
        self._is_entry_cam = camera_id in ENTRY_CAMERAS
        self._is_billing_cam = camera_id in BILLING_CAMERAS
        self._emitted_entry: set = set()
        self._emitted_exit: set = set()
        self._active_trackers_prev: set = set()

        # For billing queue tracking
        self._current_queue_depth = 0

        # For POS-based billing abandon detection (loaded separately)
        self._pos_transactions: List[dict] = []

    def load_pos_transactions(self, pos_path: str):
        """Load POS data for billing abandon detection."""
        import csv
        transactions = []
        with open(pos_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.strptime(
                        f"{row['order_date']} {row['order_time']}",
                        "%d-%m-%Y %H:%M:%S"
                    ).replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
                    transactions.append({
                        "order_id": row["order_id"],
                        "timestamp": ts,
                        "amount": float(row["GMV"] or 0),
                    })
                except Exception:
                    continue
        self._pos_transactions = sorted(transactions, key=lambda x: x["timestamp"])
        logger.info(f"Loaded {len(self._pos_transactions)} POS transactions")

    def _frame_timestamp(self, frame_idx: int, fps: float) -> datetime:
        offset_sec = frame_idx / fps
        return self.clip_start_time + timedelta(seconds=offset_sec)

    def _get_sku_zone(self, zone_id: Optional[str]) -> Optional[str]:
        if zone_id is None:
            return None
        for zone in self.layout.get("zones", []):
            if zone["zone_id"] == zone_id:
                return zone.get("sku_zone")
        return None

    def _is_billing_zone(self, zone_id: Optional[str]) -> bool:
        if zone_id is None:
            return False
        for zone in self.layout.get("zones", []):
            if zone["zone_id"] == zone_id:
                return zone.get("is_billing", False)
        return False

    def _was_purchase_made(self, billing_exit_ts: datetime, window_min: float = 5.0) -> bool:
        window_sec = window_min * 60
        for txn in self._pos_transactions:
            delta = (txn["timestamp"] - billing_exit_ts).total_seconds()
            if 0 <= delta <= window_sec:
                return True
        return False

    def process(self):
        """Main processing loop."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            sys.exit(1)

        if not os.path.exists(self.clip_path):
            logger.error(f"Clip not found: {self.clip_path}")
            return

        model = YOLO("yolov8n.pt")  # Nano model for speed; swap to yolov8s for accuracy
        logger.info(f"Processing {self.clip_path} with camera {self.camera_id}")

        cap = cv2.VideoCapture(self.clip_path)
        if not cap.isOpened():
            logger.error(f"Cannot open {self.clip_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"FPS={fps:.1f}, total_frames={total_frames}, clip_start={self.clip_start_time}")

        frame_idx = 0
        # Track zone entry times per visitor_id for ZONE_DWELL emission
        zone_entry_times: Dict[Tuple[str, str], float] = {}
        # Track billing zone entry times per visitor_id
        billing_entry: Dict[str, datetime] = {}

        # Track which tracker IDs were seen in previous iteration
        prev_tracker_ids: set = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            frame_ts = self._frame_timestamp(frame_idx, fps)
            frame_ts_unix = frame_ts.timestamp()
            frame_h, frame_w = frame.shape[:2]

            # Run YOLO with ByteTrack (class 0 = person)
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=MIN_CONF,
                iou=0.5,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            current_tracker_ids: set = set()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                track_ids = boxes.id

                if track_ids is None:
                    continue

                # Count people in billing zone for queue depth
                billing_count = 0

                for box, track_id_t in zip(boxes, track_ids):
                    track_id = int(track_id_t.item())
                    current_tracker_ids.add(track_id)
                    conf = float(box.conf.item())

                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    cx_norm = cx / frame_w
                    cy_norm = cy / frame_h

                    # Crop for appearance features
                    crop = frame[
                        max(0, int(y1)) : min(frame_h, int(y2)),
                        max(0, int(x1)) : min(frame_w, int(x2)),
                    ]

                    # Zone assignment
                    zone_id = self._zone_assigner(cx, cy, frame_h, frame_w)

                    # Register with Re-ID tracker
                    visitor_id, is_new, is_reentry = self.reid_tracker.register_detection(
                        track_id,
                        frame_ts_unix,
                        zone_id,
                        frame_crop=crop if crop.size > 0 else None,
                        bbox_center=(cx_norm, cy_norm),
                        confidence=conf,
                    )

                    state = self.reid_tracker.get_state(track_id)
                    if state is None:
                        continue

                    # Staff classification (periodic re-evaluation)
                    if frame_idx % (FRAME_SKIP * 30) == 0:
                        state.is_staff = self.reid_tracker.classify_staff(state)

                    is_staff = state.is_staff
                    sku_zone = self._get_sku_zone(zone_id)
                    seq = state.session_seq

                    # --- ENTRY CAMERA LOGIC ---
                    if self._is_entry_cam:
                        direction = self.dir_tracker.update(track_id, cy_norm, frame_h)

                        if direction == "ENTRY" and visitor_id not in self._emitted_entry:
                            self._emitted_entry.add(visitor_id)
                            seq = self.reid_tracker.increment_session_seq(track_id)
                            evt = self.emitter.make_event(
                                camera_id=self.camera_id,
                                visitor_id=visitor_id,
                                event_type="REENTRY" if is_reentry else "ENTRY",
                                timestamp=frame_ts,
                                zone_id=None,
                                dwell_ms=0,
                                is_staff=is_staff,
                                confidence=conf,
                                session_seq=seq,
                            )
                            self.emitter.emit(evt)
                            logger.debug(f"{'REENTRY' if is_reentry else 'ENTRY'}: {visitor_id}")

                        elif direction == "EXIT" and visitor_id not in self._emitted_exit:
                            self._emitted_exit.add(visitor_id)
                            seq = self.reid_tracker.increment_session_seq(track_id)
                            evt = self.emitter.make_event(
                                camera_id=self.camera_id,
                                visitor_id=visitor_id,
                                event_type="EXIT",
                                timestamp=frame_ts,
                                zone_id=None,
                                dwell_ms=0,
                                is_staff=is_staff,
                                confidence=conf,
                                session_seq=seq,
                            )
                            self.emitter.emit(evt)
                            self.reid_tracker.mark_exit(track_id, frame_ts_unix)
                            logger.debug(f"EXIT: {visitor_id}")

                    # --- ZONE EVENTS ---
                    zone_key = (visitor_id, zone_id)
                    if zone_id and zone_key not in zone_entry_times:
                        # New zone entry
                        zone_entry_times[zone_key] = frame_ts_unix
                        seq = self.reid_tracker.increment_session_seq(track_id)
                        evt = self.emitter.make_event(
                            camera_id=self.camera_id,
                            visitor_id=visitor_id,
                            event_type="ZONE_ENTER",
                            timestamp=frame_ts,
                            zone_id=zone_id,
                            dwell_ms=0,
                            is_staff=is_staff,
                            confidence=conf,
                            sku_zone=sku_zone,
                            session_seq=seq,
                        )
                        self.emitter.emit(evt)

                        # Billing zone join detection
                        if self._is_billing_zone(zone_id):
                            billing_entry[visitor_id] = frame_ts
                            if self._current_queue_depth > 0:
                                seq = self.reid_tracker.increment_session_seq(track_id)
                                evt = self.emitter.make_event(
                                    camera_id=self.camera_id,
                                    visitor_id=visitor_id,
                                    event_type="BILLING_QUEUE_JOIN",
                                    timestamp=frame_ts,
                                    zone_id=zone_id,
                                    dwell_ms=0,
                                    is_staff=is_staff,
                                    confidence=conf,
                                    queue_depth=self._current_queue_depth,
                                    sku_zone=sku_zone,
                                    session_seq=seq,
                                )
                                self.emitter.emit(evt)

                    # Dwell events (every 30s in zone)
                    if zone_id and self.reid_tracker.should_emit_dwell(
                        visitor_id, zone_id, frame_ts_unix
                    ):
                        entry_ts = zone_entry_times.get(zone_key, frame_ts_unix)
                        dwell_ms = int((frame_ts_unix - entry_ts) * 1000)
                        seq = self.reid_tracker.increment_session_seq(track_id)
                        evt = self.emitter.make_event(
                            camera_id=self.camera_id,
                            visitor_id=visitor_id,
                            event_type="ZONE_DWELL",
                            timestamp=frame_ts,
                            zone_id=zone_id,
                            dwell_ms=dwell_ms,
                            is_staff=is_staff,
                            confidence=conf,
                            sku_zone=sku_zone,
                            session_seq=seq,
                        )
                        self.emitter.emit(evt)

                    # Count for billing queue
                    if self._is_billing_cam and self._is_billing_zone(zone_id):
                        billing_count += 1

                # Update queue depth
                if self._is_billing_cam:
                    self._current_queue_depth = max(0, billing_count - 1)

            # --- Detect disappeared trackers → ZONE_EXIT / EXIT events ---
            disappeared = prev_tracker_ids - current_tracker_ids
            for tid in disappeared:
                state = self.reid_tracker.get_state(tid)
                if state is None:
                    continue

                visitor_id = state.visitor_id
                zone_id = state.current_zone
                is_staff = state.is_staff
                seq = self.reid_tracker.increment_session_seq(tid)

                if zone_id:
                    entry_ts = zone_entry_times.get((visitor_id, zone_id), frame_ts_unix)
                    dwell_ms = int((frame_ts_unix - entry_ts) * 1000)
                    evt = self.emitter.make_event(
                        camera_id=self.camera_id,
                        visitor_id=visitor_id,
                        event_type="ZONE_EXIT",
                        timestamp=frame_ts,
                        zone_id=zone_id,
                        dwell_ms=dwell_ms,
                        is_staff=is_staff,
                        confidence=0.8,
                        sku_zone=self._get_sku_zone(zone_id),
                        session_seq=seq,
                    )
                    self.emitter.emit(evt)
                    zone_entry_times.pop((visitor_id, zone_id), None)

                    # Billing abandon detection
                    if self._is_billing_zone(zone_id) and visitor_id in billing_entry:
                        billing_enter_ts = billing_entry.pop(visitor_id)
                        if not self._was_purchase_made(frame_ts):
                            seq = self.reid_tracker.increment_session_seq(tid)
                            evt = self.emitter.make_event(
                                camera_id=self.camera_id,
                                visitor_id=visitor_id,
                                event_type="BILLING_QUEUE_ABANDON",
                                timestamp=frame_ts,
                                zone_id=zone_id,
                                dwell_ms=int((frame_ts - billing_enter_ts).total_seconds() * 1000),
                                is_staff=is_staff,
                                confidence=0.8,
                                sku_zone=self._get_sku_zone(zone_id),
                                session_seq=seq,
                            )
                            self.emitter.emit(evt)

                self.reid_tracker.mark_exit(tid, frame_ts_unix)
                self.dir_tracker.remove(tid)

            prev_tracker_ids = current_tracker_ids

            if frame_idx % 150 == 0:
                elapsed_min = frame_idx / fps / 60
                logger.info(
                    f"[{self.camera_id}] Processed {frame_idx}/{total_frames} frames "
                    f"({elapsed_min:.1f} min clip time)"
                )

        cap.release()
        logger.info(f"[{self.camera_id}] Clip processing complete")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def load_layout(layout_path: Path) -> dict:
    with open(layout_path) as f:
        return json.load(f)


def find_camera_config(layout: dict, camera_id: str) -> Optional[dict]:
    for cam in layout.get("cameras", []):
        if cam["camera_id"] == camera_id:
            return cam
    return None


def parse_clip_start(cam_config: dict) -> datetime:
    ts_str = cam_config.get("clip_start_time", "2026-04-10T12:00:00+05:30")
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        return datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def main():
    parser = argparse.ArgumentParser(description="CCTV Detection Pipeline")
    parser.add_argument("--clip", type=str, help="Path to a single video clip")
    parser.add_argument("--camera", type=str, help="Camera ID for the clip")
    parser.add_argument("--all", action="store_true", help="Process all cameras")
    parser.add_argument(
        "--footage-dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "CCTV Footage"),
        help="Directory containing footage files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent.parent / "events.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--pos",
        type=str,
        default=str(
            Path(__file__).parent.parent.parent / "Brigade_Bangalore_10_April_26 (1)bc6219c.csv"
        ),
        help="POS transactions CSV",
    )
    args = parser.parse_args()

    layout = load_layout(LAYOUT_FILE)
    store_id = layout["store_id"]

    # Clear output file for fresh run
    output_path = args.output
    open(output_path, "w").close()

    emitter = EventEmitter(output_path, store_id)

    cameras_to_process = []
    if args.all:
        for cam in layout["cameras"]:
            clip_path = os.path.join(args.footage_dir, cam["file"])
            cameras_to_process.append((clip_path, cam["camera_id"], cam))
    elif args.clip and args.camera:
        cam_config = find_camera_config(layout, args.camera)
        if cam_config is None:
            logger.error(f"Unknown camera_id: {args.camera}")
            sys.exit(1)
        cameras_to_process.append((args.clip, args.camera, cam_config))
    else:
        parser.print_help()
        sys.exit(1)

    for clip_path, camera_id, cam_config in cameras_to_process:
        clip_start = parse_clip_start(cam_config)
        processor = ClipProcessor(
            clip_path=clip_path,
            camera_id=camera_id,
            emitter=emitter,
            clip_start_time=clip_start,
            layout=layout,
        )
        if os.path.exists(args.pos):
            processor.load_pos_transactions(args.pos)
        processor.process()

    emitter.close()
    logger.info(f"All done. Events written to {output_path}")


if __name__ == "__main__":
    main()
