"""
Re-ID and tracking logic.

Handles:
- Per-tracker visitor_id assignment
- Cross-frame tracking state
- Re-entry detection (same person reappears after EXIT)
- Appearance-based Re-ID using color histogram
- Staff classification using movement patterns
"""

import hashlib
import time
import logging
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Re-entry window: if same appearance reappears within 15 minutes, it is a re-entry
REENTRY_WINDOW_SEC = 900

# Staff detection: if a tracker is seen 3+ times across multiple zone visits
# and spends unusually short time (staff moves quickly), flag as staff
STAFF_MIN_ZONE_VISITS = 5
STAFF_MAX_AVG_DWELL_PER_ZONE_SEC = 45.0
STAFF_MIN_TOTAL_APPEARANCES = 8

# Appearance similarity threshold (cosine similarity of color histograms)
APPEARANCE_SIMILARITY_THRESHOLD = 0.80


@dataclass
class TrackerState:
    tracker_id: int
    visitor_id: str
    first_seen_ts: float
    last_seen_ts: float
    current_zone: Optional[str]
    zone_entry_ts: float
    zone_history: List[Tuple[str, float, float]]  # (zone_id, enter_ts, exit_ts)
    appearance_hist: Optional[np.ndarray]  # color histogram for Re-ID
    is_staff: bool = False
    is_active: bool = True
    session_seq: int = 0
    entry_position: Optional[Tuple[float, float]] = None
    exit_seen: bool = False
    reentry_count: int = 0


@dataclass
class ExitedVisitor:
    visitor_id: str
    exit_ts: float
    appearance_hist: Optional[np.ndarray]
    last_zone: Optional[str]
    entry_position: Optional[Tuple[float, float]]


class ReIDTracker:
    """
    Maps tracker IDs (from ByteTrack / YOLO tracker) to stable visitor_ids.
    Handles re-entry detection via appearance matching.
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        # Active tracker states keyed by tracker_id
        self.active: Dict[int, TrackerState] = {}
        # Recently exited visitors available for re-ID matching
        self.exited: List[ExitedVisitor] = []
        # Mapping from tracker_id to visitor_id for fast lookup
        self._tid_to_vid: Dict[int, str] = {}
        # Zone dwell tracking: (visitor_id, zone_id) → last dwell event ts
        self.last_dwell_event: Dict[Tuple[str, str], float] = {}

    def _make_visitor_id(self, tracker_id: int) -> str:
        token = f"{self.camera_id}_{tracker_id}_{time.time_ns()}"
        h = hashlib.md5(token.encode()).hexdigest()[:6]
        return f"VIS_{h}"

    def _compute_histogram(self, frame_crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Compute a compact color histogram for appearance matching."""
        if frame_crop is None or frame_crop.size == 0:
            return None
        try:
            import cv2
            hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
            hist = np.concatenate([h_hist, s_hist])
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
            return hist
        except Exception:
            return None

    def _appearance_similarity(
        self, h1: Optional[np.ndarray], h2: Optional[np.ndarray]
    ) -> float:
        if h1 is None or h2 is None:
            return 0.0
        return float(np.dot(h1, h2))

    def _find_reid_match(
        self,
        appearance_hist: Optional[np.ndarray],
        entry_position: Tuple[float, float],
        current_ts: float,
    ) -> Optional[ExitedVisitor]:
        """Try to match a new detection to a recently exited visitor (re-entry)."""
        best_score = APPEARANCE_SIMILARITY_THRESHOLD
        best_match = None

        for ev in self.exited:
            if current_ts - ev.exit_ts > REENTRY_WINDOW_SEC:
                continue
            sim = self._appearance_similarity(appearance_hist, ev.appearance_hist)
            if sim > best_score:
                best_score = sim
                best_match = ev

        return best_match

    def register_detection(
        self,
        tracker_id: int,
        timestamp: float,
        zone_id: Optional[str],
        frame_crop: Optional[np.ndarray] = None,
        bbox_center: Optional[Tuple[float, float]] = None,
        confidence: float = 1.0,
    ) -> Tuple[str, bool, bool]:
        """
        Register a detection for a tracker.
        Returns (visitor_id, is_new_entry, is_reentry).
        """
        if tracker_id in self.active:
            state = self.active[tracker_id]
            state.last_seen_ts = timestamp
            if zone_id and zone_id != state.current_zone:
                # Zone change — close old zone
                if state.current_zone:
                    state.zone_history.append(
                        (state.current_zone, state.zone_entry_ts, timestamp)
                    )
                state.current_zone = zone_id
                state.zone_entry_ts = timestamp
            return state.visitor_id, False, False

        # New tracker ID — check if it is a re-entry
        appearance_hist = self._compute_histogram(frame_crop)
        is_reentry = False
        visitor_id = None

        if bbox_center:
            match = self._find_reid_match(appearance_hist, bbox_center, timestamp)
            if match:
                visitor_id = match.visitor_id
                is_reentry = True
                self.exited = [e for e in self.exited if e.visitor_id != visitor_id]
                logger.debug(f"Re-ID match: tracker {tracker_id} → {visitor_id}")

        if visitor_id is None:
            visitor_id = self._make_visitor_id(tracker_id)

        state = TrackerState(
            tracker_id=tracker_id,
            visitor_id=visitor_id,
            first_seen_ts=timestamp,
            last_seen_ts=timestamp,
            current_zone=zone_id,
            zone_entry_ts=timestamp,
            zone_history=[],
            appearance_hist=appearance_hist,
            entry_position=bbox_center,
            reentry_count=1 if is_reentry else 0,
        )
        self.active[tracker_id] = state
        self._tid_to_vid[tracker_id] = visitor_id
        return visitor_id, True, is_reentry

    def mark_exit(self, tracker_id: int, timestamp: float) -> Optional[TrackerState]:
        """Mark a tracker as exited. Returns the closed state."""
        if tracker_id not in self.active:
            return None
        state = self.active.pop(tracker_id)
        state.is_active = False
        state.exit_seen = True
        if state.current_zone:
            state.zone_history.append((state.current_zone, state.zone_entry_ts, timestamp))
            state.current_zone = None

        self.exited.append(
            ExitedVisitor(
                visitor_id=state.visitor_id,
                exit_ts=timestamp,
                appearance_hist=state.appearance_hist,
                last_zone=state.zone_history[-1][0] if state.zone_history else None,
                entry_position=state.entry_position,
            )
        )
        # Prune old exited visitors
        self.exited = [
            e for e in self.exited if timestamp - e.exit_ts <= REENTRY_WINDOW_SEC
        ]
        return state

    def get_state(self, tracker_id: int) -> Optional[TrackerState]:
        return self.active.get(tracker_id)

    def classify_staff(self, state: TrackerState) -> bool:
        """
        Heuristic staff classification:
        - High total zone visits (staff moves around a lot)
        - Short average dwell time per zone
        - Long total presence time
        """
        total_zone_visits = len(state.zone_history)
        total_duration = state.last_seen_ts - state.first_seen_ts

        if total_zone_visits < STAFF_MIN_ZONE_VISITS:
            return False

        total_dwell = sum(
            (exit_t - enter_t) for _, enter_t, exit_t in state.zone_history
        )
        avg_dwell = total_dwell / max(total_zone_visits, 1)

        if (
            avg_dwell < STAFF_MAX_AVG_DWELL_PER_ZONE_SEC
            and total_zone_visits >= STAFF_MIN_ZONE_VISITS
            and total_duration > 300  # at least 5 minutes of presence
        ):
            return True
        return False

    def should_emit_dwell(
        self, visitor_id: str, zone_id: str, current_ts: float
    ) -> bool:
        """Returns True if a ZONE_DWELL event should be emitted (every 30s)."""
        key = (visitor_id, zone_id)
        last = self.last_dwell_event.get(key, 0.0)
        if current_ts - last >= 30.0:
            self.last_dwell_event[key] = current_ts
            return True
        return False

    def increment_session_seq(self, tracker_id: int) -> int:
        if tracker_id in self.active:
            self.active[tracker_id].session_seq += 1
            return self.active[tracker_id].session_seq
        return 0
