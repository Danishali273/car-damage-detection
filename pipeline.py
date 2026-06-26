"""
pipeline.py — Production-Grade Car Damage Detection Pipeline
=============================================================
Integrates three models in a single, modular pipeline:
  1. Direction Classifier  (best_car_angle.pt)   — YOLOv8 classify
  2. Parts Segmenter       (best_car_part.pt)    — YOLOv8 seg
  3. Damage Detector       (best_damage_type.pt) — YOLOv8 detect

Key architectural features
--------------------------
  • Coordinate Transformation  — camera-view → car-centric labels
  • Context-Aware Part Filtering — only relevant parts per view are processed
  • Temporal Aggregation (DamageRegistry) — tracks damage per Track-ID across frames
  • Voting / Threshold logic — suppresses single-frame noise ("flickering")
  • Hierarchical Damage Localisation — "Part + Direction" compound labels
  • Flicker Suppression — direction uncertainty is smoothed with a rolling buffer

Usage
-----
  python pipeline.py testvideo2.mp4
  python pipeline.py testvideo2.mp4 --output out.mp4 --frame-skip 2 --preview
  python pipeline.py testvideo2.mp4 --report report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

import cv2
import numpy as np
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION — edit thresholds here without touching pipeline logic
# ══════════════════════════════════════════════════════════════════════════════

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_ANGLE_PATH  = "models/best_car_angle.pt"
MODEL_PARTS_PATH  = "models/best_car_part.pt"
MODEL_DAMAGE_PATH = "models/best_damage_type.pt"

# ── Perspective map (camera-view → car-centric) ───────────────────────────────
# The camera captures a mirror-image of the car's true side.
# e.g., filming the car's LEFT side from the right means you are on its right.
PERSPECTIVE_MAP: Dict[str, str] = {
    "front-right": "front-left-side",
    "front-left":  "front-right-side",
    "back-right":  "back-left-side",
    "back-left":   "back-right-side",
    "side-right":  "left-side",
    "side-left":   "right-side",
    "front":       "front",
    "back":        "back",
}

# ── Context-aware part list per car-centric direction ─────────────────────────
CAR_PARTS_MAP: Dict[str, List[str]] = {
    "front": [
        "Front-bumper", "Grille", "Headlight", "Hood",
        "License-plate", "Windshield",
    ],
    "back": [
        "Back-bumper", "Trunk", "Tail-light", "Back-windshield",
    ],
    "left-side": [
        "Front-door", "Back-door", "Front-wheel", "Back-wheel",
        "Fender", "Quarter-panel", "Mirror", "Rocker-panel",
    ],
    "right-side": [
        "Front-door", "Back-door", "Front-wheel", "Back-wheel",
        "Fender", "Quarter-panel", "Mirror", "Rocker-panel",
    ],
    "front-left-side": [
        "Front-bumper", "Fender", "Front-door", "Front-wheel",
        "Mirror", "Hood", "Headlight", "Windshield",
    ],
    "front-right-side": [
        "Front-bumper", "Fender", "Front-door", "Front-wheel",
        "Mirror", "Hood", "Headlight", "Windshield",
    ],
    "back-left-side": [
        "Back-bumper", "Quarter-panel", "Back-door",
        "Back-wheel", "Tail-light", "Back-windshield",
    ],
    "back-right-side": [
        "Back-bumper", "Quarter-panel", "Back-door",
        "Back-wheel", "Tail-light", "Back-windshield",
    ],
}

# ── Per-class damage thresholds ───────────────────────────────────────────────
DAMAGE_THRESHOLDS: Dict[str, float] = {
    "dent":         0.50,
    "glass_break":  0.50,
    "scratch":      0.50,
    "smash":        0.70,
    "crack":        0.50,
    "broken_light": 0.50,
    "flat_tire":    0.90,
}

# ── Per-part segmentation thresholds ─────────────────────────────────────────
# NOTE: Keys must exactly match the part names used in CAR_PARTS_MAP and
# returned by the parts segmentation model.  Entries not in CAR_PARTS_MAP
# (Back-window, Front-window, Roof) are retained because the model may
# still detect them; they will simply be filtered by the context-aware list.
PART_THRESHOLDS: Dict[str, float] = {k: 0.50 for k in [
    "Back-bumper", "Back-door", "Back-wheel", "Back-window",
    "Back-windshield", "Fender", "Front-bumper", "Front-door",
    "Front-wheel", "Front-window", "Grille", "Headlight", "Hood",
    "License-plate", "Mirror", "Quarter-panel", "Rocker-panel",
    "Roof", "Tail-light", "Trunk", "Windshield",
]}

# ── Which damage types are physically possible on each part ───────────────────
# Covers every part that appears in CAR_PARTS_MAP plus glass/roof variants
# that the segmentation model may detect outside of strict context filtering.
PART_DAMAGE_MAP: Dict[str, List[str]] = {
    # Wheels
    "Back-wheel":      ["flat_tire"],
    "Front-wheel":     ["flat_tire"],
    # Glass surfaces
    "Back-window":     ["glass_break"],
    "Back-windshield": ["glass_break"],
    "Front-window":    ["glass_break"],
    "Windshield":      ["glass_break"],
    # Lights
    "Headlight":       ["broken_light"],
    "Tail-light":      ["broken_light"],
    # Small exterior parts
    "Mirror":          ["crack", "scratch"],
    "Grille":          ["crack"],
    "License-plate":   ["dent", "scratch"],
    # Bumpers
    "Front-bumper":    ["dent", "scratch", "crack"],
    "Back-bumper":     ["dent", "scratch", "crack"],
    # Body panels
    "Hood":            ["dent", "scratch", "smash", "crack"],
    "Trunk":           ["dent", "scratch", "smash", "crack"],
    "Roof":            ["dent", "scratch", "smash", "crack"],
    "Fender":          ["dent", "scratch", "smash", "crack"],
    "Front-door":      ["dent", "scratch", "smash", "crack"],
    "Back-door":       ["dent", "scratch", "smash", "crack"],
    "Quarter-panel":   ["dent", "scratch", "smash", "crack"],
    "Rocker-panel":    ["dent", "scratch", "smash", "crack"],
}
_BODY_PANEL_DEFAULT = ["dent", "scratch", "smash", "crack"]

# ── DamageRegistry voting parameters ─────────────────────────────────────────
# Lowered defaults — 2 votes + 15 % ratio works well for short walkaround clips.
# Raise these values to reduce false positives on longer recordings.
REGISTRY_MIN_VOTES      = 3    # minimum frames a damage must appear to be "confirmed"
REGISTRY_MIN_VOTE_RATIO = 0.25  # damage seen in >= 15 % of frames it was observable
DIRECTION_BUFFER_LEN    = 5    # rolling window for direction flicker suppression
DIRECTION_CONF_THRESHOLD = 0.60  # minimum classifier confidence to accept a direction

# ── Visual / HUD colours ──────────────────────────────────────────────────────
PALETTE = [
    (230,  25,  75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145,  30, 180), (70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,  0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (0,   0, 128), (128, 128, 128),
]

# ── Supported image extensions (auto-detected in CLI) ────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def part_color(part_name: str) -> Tuple[int, int, int]:
    """Return a consistent BGR colour for a given part label.

    Uses MD5 (first 4 bytes → unsigned int) so the mapping is stable
    across Python interpreter runs regardless of PYTHONHASHSEED.
    """
    digest = hashlib.md5(part_name.encode()).digest()
    idx = int.from_bytes(digest[:4], "big") % len(PALETTE)
    return PALETTE[idx]


def get_allowed_damage(part_name: str) -> List[str]:
    """Return damage types that are physically possible on this part."""
    return PART_DAMAGE_MAP.get(part_name, _BODY_PANEL_DEFAULT)


def passes_damage_threshold(damage_class: str, conf: float) -> bool:
    return conf >= DAMAGE_THRESHOLDS.get(damage_class, 0.40)


def passes_part_threshold(part_name: str, conf: float) -> bool:
    return conf >= PART_THRESHOLDS.get(part_name, 0.50)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  COORDINATE TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

class PerspectiveTransformer:
    """
    Translates raw camera-view labels from the direction classifier
    into car-centric direction labels.

    Background
    ----------
    The camera sees the car as a mirror image of its own coordinate
    system.  E.g. when the camera is to the car's LEFT, the car appears
    to be on the RIGHT of the frame, so the classifier outputs 'side-right'
    but the car-centric direction is actually 'left-side'.
    """

    def __init__(self, mapping: Dict[str, str] = PERSPECTIVE_MAP) -> None:
        self._map = mapping

    def transform(self, raw_label: str) -> str:
        """
        Convert a camera-view label to its car-centric equivalent.
        Returns the raw label unchanged if it isn't in the map
        (acts as a safe passthrough for unexpected classes).
        """
        return self._map.get(raw_label, raw_label)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DIRECTION FLICKER SUPPRESSOR
# ══════════════════════════════════════════════════════════════════════════════

class DirectionBuffer:
    """
    Smooths rapid direction changes caused by:
      - The car rotating during a walkaround
      - Low-confidence classification frames
      - Motion blur degrading the angle classifier

    Strategy
    --------
    Maintains a rolling window of the last N car-centric direction labels.
    The *mode* (most frequent label) over the window is used as the current
    stable direction.  A frame is only accepted into the buffer when the
    classifier's top-1 confidence is above DIRECTION_CONF_THRESHOLD.

    If the classifier is uncertain (conf < threshold), the previous stable
    direction is reused — this is the "flicker suppression" mechanism.

    Low-confidence fallback
    -----------------------
    If confidence has been below the threshold for `max_low_conf_streak`
    consecutive frames (default: 10), the raw direction label is accepted
    regardless — preventing the pipeline from silently producing no output
    on low-quality or heavily compressed footage.
    """

    def __init__(
        self,
        window: int = DIRECTION_BUFFER_LEN,
        conf_min: float = DIRECTION_CONF_THRESHOLD,
        max_low_conf_streak: int = 10,
    ) -> None:
        self._buf: deque[str] = deque(maxlen=window)
        self._conf_min = conf_min
        self._max_streak = max_low_conf_streak
        self._low_conf_streak: int = 0
        self._last_stable: Optional[str] = None

    def update(self, direction: str, conf: float) -> Optional[str]:
        """
        Feed a new raw observation.

        Parameters
        ----------
        direction : car-centric direction label
        conf      : classifier confidence for the top-1 prediction

        Returns
        -------
        Stable direction string, or None if the buffer is still warming up
        and no stable direction has been seen yet.
        """
        if conf >= self._conf_min:
            # High-confidence frame → accept it and reset the low-conf streak.
            self._buf.append(direction)
            self._low_conf_streak = 0
        else:
            self._low_conf_streak += 1
            if self._low_conf_streak >= self._max_streak:
                # Too many consecutive low-confidence frames — accept the raw
                # direction as a fallback so the pipeline doesn't silently
                # skip an entire low-quality video segment.
                log.warning(
                    "DirectionBuffer: %d consecutive low-conf frames "
                    "(conf<%.2f); accepting '%s' as fallback.",
                    self._low_conf_streak, self._conf_min, direction,
                )
                self._buf.append(direction)
                self._low_conf_streak = 0
            # else: skip this frame, keep the window unchanged
            #       (equivalent to "holding the last known good direction")

        if self._buf:
            # Mode vote over the rolling window
            counts: Dict[str, int] = defaultdict(int)
            for d in self._buf:
                counts[d] += 1
            self._last_stable = max(counts, key=counts.__getitem__)

        return self._last_stable

    @property
    def stable_direction(self) -> Optional[str]:
        return self._last_stable


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DAMAGE LOCALIZATION RESOLVER
# ══════════════════════════════════════════════════════════════════════════════

def resolve_damage_location(part_name: str, car_direction: str) -> str:
    """
    Produce a human-readable compound damage location.

    Hierarchical logic
    ------------------
    Raw part + raw direction → sanitised compound label.

    Examples
    --------
    ("Front-door", "front-left-side")  → "Front Left Side Front Door"
    ("Hood",        "front")            → "Front Hood"
    ("Back-bumper", "back")             → "Back Bumper"   ← deduplication

    Redundant direction words that already appear in the part name are
    removed from the prefix so we never produce strings like
    "Back Back-Bumper".
    """
    direction_prefix = car_direction.replace("-", " ").title()  # e.g. "Back"
    part_display     = part_name.replace("-", " ")              # e.g. "Back Bumper"

    # Drop direction words already present in the part name (case-insensitive)
    prefix_words = direction_prefix.split()
    part_words   = {w.lower() for w in part_display.split()}
    filtered     = [w for w in prefix_words if w.lower() not in part_words]
    prefix       = " ".join(filtered)

    return f"{prefix} {part_display}".strip() if prefix else part_display


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DAMAGE REGISTRY  (Temporal Aggregation / Voting)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DamageObservation:
    """A single per-frame observation of a damage event on a part."""
    damage_type:  str
    confidence:   float
    location:     str        # e.g. "Front-Left-Side Front-Door"
    car_direction: str
    frame_index:  int


@dataclass
class PartRecord:
    """
    Aggregated record for a single (track_id, part_name, car_direction) triplet.

    Using car_direction as part of the key ensures that a physically large part
    (e.g. "Front-bumper", "Front-door") observed from two distinct car-centric
    directions (e.g. "front" vs "front-left-side") is treated as a separate
    entity in the registry.  This prevents the old winner-takes-all overwrite
    that occurred when all views of the same part were collapsed into one record.

    Fields
    ------
    observations       : all raw per-frame damage observations
    total_frames_seen  : how many frames this part was detected in *from this direction*
    confirmed_damages  : dict mapping damage_type → best confidence
                         (populated after finalize() is called)
    """
    part_name:          str
    track_id:           int
    car_direction:      str
    observations:       List[DamageObservation] = field(default_factory=list)
    total_frames_seen:  int = 0
    confirmed_damages:  Dict[str, float] = field(default_factory=dict)

    def add_observation(self, obs: DamageObservation) -> None:
        # NOTE: do NOT increment total_frames_seen here.
        # mark_part_seen() is always called first (even on frames with no damage),
        # so the counter is already correct.  Incrementing again here would
        # double-count damaged frames and artificially halve the vote ratio.
        self.observations.append(obs)

    def finalize(
        self,
        min_votes: int = REGISTRY_MIN_VOTES,
        min_ratio: float = REGISTRY_MIN_VOTE_RATIO,
    ) -> None:
        """
        Apply voting logic to determine which damage types are truly present.

        A damage type is "confirmed" if:
          1. It was observed in at least `min_votes` frames, AND
          2. It appeared in at least `min_ratio` fraction of total frames seen.

        This dual requirement prevents both:
          - Single-frame false positives (min_votes guard)
          - Detections on parts that appeared briefly (min_ratio guard)
        """
        vote_counts: Dict[str, int] = defaultdict(int)
        best_conf:   Dict[str, float] = defaultdict(float)

        for obs in self.observations:
            vote_counts[obs.damage_type] += 1
            if obs.confidence > best_conf[obs.damage_type]:
                best_conf[obs.damage_type] = obs.confidence

        self.confirmed_damages = {
            d_type: best_conf[d_type]
            for d_type, votes in vote_counts.items()
            if votes >= min_votes
            and (votes / max(self.total_frames_seen, 1)) >= min_ratio
        }


class DamageRegistry:
    """
    Central store that accumulates damage evidence across all video frames.

    Why a registry?
    ---------------
    A single car panel (e.g. "Front-bumper") may be visible from multiple
    camera angles (front, front-left-side, front-right-side).  The registry
    now uses a *three-part key* ``(track_id, part_name, car_direction)`` so
    that the same physical surface viewed from different car-centric directions
    is tracked as a separate entity.  This prevents the old winner-takes-all
    behaviour where one directional observation silently overwrote another,
    and ensures that both "Left-Side Front-Door" and "Right-Side Front-Door"
    can appear as distinct entries in the final damage report.

    Track ID strategy
    -----------------
    For ByteTrack-style pipelines, pass the YOLO-assigned track_id directly.
    For single-car scenarios without tracking, use track_id=0 for all parts;
    the registry still correctly aggregates per (part, direction) across frames.
    """

    def __init__(
        self,
        min_votes: int  = REGISTRY_MIN_VOTES,
        min_ratio: float = REGISTRY_MIN_VOTE_RATIO,
    ) -> None:
        self._min_votes = min_votes
        self._min_ratio = min_ratio
        # key: (track_id, part_name, car_direction)  →  PartRecord
        self._records: Dict[Tuple[int, str, str], PartRecord] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        track_id:     int,
        part_name:    str,
        damage_type:  str,
        confidence:   float,
        car_direction: str,
        frame_index:  int,
    ) -> None:
        """Record a single damage observation from one frame.

        The registry key is now ``(track_id, part_name, car_direction)``.
        Damage seen on the same part from two different car-centric directions
        will accumulate in *separate* PartRecords and appear as distinct
        entries in the final damage report.
        """
        key = (track_id, part_name, car_direction)
        if key not in self._records:
            self._records[key] = PartRecord(
                part_name=part_name, track_id=track_id, car_direction=car_direction
            )

        location = resolve_damage_location(part_name, car_direction)
        obs = DamageObservation(
            damage_type=damage_type,
            confidence=confidence,
            location=location,
            car_direction=car_direction,
            frame_index=frame_index,
        )
        self._records[key].add_observation(obs)

    def mark_part_seen(
        self,
        track_id:      int,
        part_name:     str,
        car_direction: str,
    ) -> None:
        """
        Increment the 'frames seen' counter for a (part, direction) pair even
        when no damage is detected on that frame.  This keeps the min_ratio
        denominator accurate per directional view, preventing a low-observation
        direction from incorrectly boosting or diluting the vote ratio of
        another direction for the same part name.
        """
        key = (track_id, part_name, car_direction)
        if key not in self._records:
            self._records[key] = PartRecord(
                part_name=part_name, track_id=track_id, car_direction=car_direction
            )
        self._records[key].total_frames_seen += 1

    # ------------------------------------------------------------------
    def finalize(self) -> List[Dict]:
        """
        Run voting logic on all records and return the final damage report.

        Because the registry key is now ``(track_id, part_name, car_direction)``,
        the same physical part observed from two distinct car-centric directions
        will produce **separate entries** in the report (e.g. the left and right
        halves of a bumper can both appear with their own damage types and
        confidence scores).

        Returns
        -------
        List of dicts, one per confirmed damaged location, sorted by
        (track_id, part_name, car_direction).  Each dict contains:
            track_id, part_name, car_direction, location, damage_type, confidence
        """
        report: List[Dict] = []

        for (track_id, part_name, car_direction), record in sorted(self._records.items()):
            record.finalize(self._min_votes, self._min_ratio)

            for d_type, best_conf in record.confirmed_damages.items():
                # Determine the most frequent location label for this damage
                # within *this directional view* of the part.
                loc_votes: Dict[str, int] = defaultdict(int)
                for obs in record.observations:
                    if obs.damage_type == d_type:
                        loc_votes[obs.location] += 1
                best_loc = max(loc_votes, key=loc_votes.__getitem__) if loc_votes else part_name

                report.append({
                    "part_name":    part_name,
                    "car_direction": car_direction,
                    "location":     best_loc,
                    "damage_type":  d_type,
                    "confidence":   round(best_conf, 4),
                })

        return report

    @staticmethod
    def format_report(report: List[Dict]) -> str:
        """
        Format a pre-finalized damage report list as a human-readable string.

        Accepts the list returned by ``finalize()`` directly, so callers that
        need both the structured data and a printable summary can call
        ``finalize()`` **once** and pass the result to both this method and
        ``json.dumps`` — avoiding a redundant second round of voting logic.
        """
        if not report:
            return "No confirmed damage detected."
        lines = ["=" * 72, "  DAMAGE REPORT", "=" * 72]
        for item in report:
            lines.append(
                f"  {item['location']:<32}  "
                f"[{item['car_direction']:<18}]  "
                f"{item['damage_type']:<14}  "
                f"conf={item['confidence']:.2f}"
            )
        lines.append("=" * 72)
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable summary of the finalized report.

        Convenience wrapper: calls ``finalize()`` then ``format_report()``.
        If you also need the structured list (e.g. to write JSON), call
        ``finalize()`` directly and pass the result to ``format_report()``
        to avoid running the voting logic twice.
        """
        return self.format_report(self.finalize())

    def debug_registry(self) -> str:
        """
        Print raw vote counts for EVERY detected (part, direction) pair
        BEFORE the voting thresholds are applied.  Use this to diagnose
        why confirmed damage is empty — you'll see if the models are
        detecting anything at all and how many votes each damage got.

        Each row now shows the car_direction alongside the part name so you
        can distinguish, e.g., a "Front-door" seen from "left-side" vs
        one seen from "front-left-side".

        Example output:
          [Track 0] Front-bumper (front)          seen=12  |  dent:  6 votes  ratio=0.50  conf=0.68  CONFIRMED
          [Track 0] Front-bumper (front-left-side) seen= 5  |  scratch: 1 votes ratio=0.20  conf=0.52  FAIL(votes<2)
          [Track 0] Hood         (front)           seen= 4  |  (no damage detected on this part)
        """
        lines = ["=" * 80, "  DEBUG — Raw Registry State (before voting)", "=" * 80]
        if not self._records:
            lines.append("  Registry is empty — no parts were detected at all.")
            lines.append("  Possible causes:")
            lines.append("    1. Direction classifier returned labels not in PERSPECTIVE_MAP")
            lines.append("    2. Parts model conf too high — try --parts-conf 0.15")
            lines.append("    3. Video is too short for the direction buffer to warm up")
            lines.append("=" * 80)
            return "\n".join(lines)

        for (track_id, part_name, car_direction), record in sorted(self._records.items()):
            vote_counts: Dict[str, int] = defaultdict(int)
            best_conf:   Dict[str, float] = defaultdict(float)
            for obs in record.observations:
                vote_counts[obs.damage_type] += 1
                if obs.confidence > best_conf[obs.damage_type]:
                    best_conf[obs.damage_type] = obs.confidence

            seen      = record.total_frames_seen
            part_key  = f"{part_name} ({car_direction})"
            if not vote_counts:
                lines.append(
                    f"  [Track {track_id:>3}] {part_key:<40} seen={seen:>3}  |  "
                    f"(no damage detected on this part)"
                )
            else:
                for d_type, votes in sorted(vote_counts.items(),
                                            key=lambda x: x[1], reverse=True):
                    ratio = votes / max(seen, 1)
                    conf  = best_conf[d_type]
                    passes_v = votes >= self._min_votes
                    passes_r = ratio >= self._min_ratio
                    verdict  = "CONFIRMED" if (passes_v and passes_r) else (
                        f"FAIL({'votes<'+str(self._min_votes) if not passes_v else ''}"
                        f"{',' if not passes_v and not passes_r else ''}"
                        f"{'ratio<'+f'{self._min_ratio:.0%}' if not passes_r else ''})"
                    )
                    lines.append(
                        f"  [Track {track_id:>3}] {part_key:<40} seen={seen:>3}  |  "
                        f"{d_type:<14} {votes:>2} votes  "
                        f"ratio={ratio:.2f}  conf={conf:.2f}  {verdict}"
                    )

        lines.append("=" * 80)
        return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# 7.  MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

class DirectionClassifier:
    """Wraps the YOLOv8 classification model for camera-angle prediction."""

    def __init__(self, model_path: str = MODEL_ANGLE_PATH) -> None:
        self.model = YOLO(model_path)
        self._transformer = PerspectiveTransformer()

    def predict(self, frame: np.ndarray) -> Tuple[str, str, float]:
        """
        Run inference on a single frame.

        Returns
        -------
        raw_label     : original classifier output  (e.g. 'front-right')
        car_direction : car-centric direction label  (e.g. 'front-left-side')
        confidence    : top-1 confidence score
        """
        results = self.model.predict(frame, verbose=False)
        probs = results[0].probs
        top1_id = probs.top1
        confidence = float(probs.top1conf)
        raw_label = results[0].names[top1_id]
        car_direction = self._transformer.transform(raw_label)
        return raw_label, car_direction, confidence


class PartsSegmenter:
    """Wraps the YOLOv8-seg model for car part segmentation."""

    def __init__(self, model_path: str = MODEL_PARTS_PATH) -> None:
        self.model = YOLO(model_path)

    def predict(
        self,
        frame: np.ndarray,
        allowed_parts: List[str],
        conf_floor: float = 0.30,
    ) -> List[Dict]:
        """
        Segment the frame and return only the parts in `allowed_parts`.

        Context-aware filtering
        -----------------------
        The `allowed_parts` list is derived from CAR_PARTS_MAP[car_direction].
        By ignoring parts that are geometrically impossible from the current
        camera angle, we avoid expensive crop-inference on irrelevant regions
        and dramatically reduce false positives.

        Returns
        -------
        List of part dicts with keys:
            part_name, conf, bbox (x1,y1,x2,y2), mask_pts, box_index
        """
        results = self.model.predict(frame, conf=conf_floor, verbose=False)
        boxes = results[0].boxes
        masks = results[0].masks

        parts: List[Dict] = []
        for i, box in enumerate(boxes):
            cls_id    = int(box.cls[0])
            part_name = self.model.names[cls_id]
            conf      = float(box.conf[0])

            # ── Context-aware filter ──────────────────────────────────────────
            if part_name not in allowed_parts:
                continue
            if not passes_part_threshold(part_name, conf):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mask_pts = None
            if masks is not None and i < len(masks.xy):
                mask_pts = masks.xy[i].astype(np.int32)

            parts.append({
                "part_name": part_name,
                "conf":      conf,
                "bbox":      (x1, y1, x2, y2),
                "mask_pts":  mask_pts,
                "box_index": i,
            })
        return parts


class DamageDetector:
    """Wraps the damage detection model and runs it on part crops."""

    def __init__(self, model_path: str = MODEL_DAMAGE_PATH) -> None:
        self.model = YOLO(model_path)

    def predict_on_crop(
        self,
        crop: np.ndarray,
        allowed_damages: List[str],
        conf_floor: float = 0.30,
    ) -> List[Tuple[str, float, Optional[np.ndarray]]]:
        """
        Run damage detection on a single part crop.

        Parameters
        ----------
        crop            : cropped BGR image of the part
        allowed_damages : damage types valid for this part (from PART_DAMAGE_MAP)
        conf_floor      : raw model confidence floor

        Returns
        -------
        List of ``(damage_type, confidence, mask_pts)`` tuples — one entry per
        distinct confirmed damage type found on the crop, sorted by descending
        confidence.  Returns an empty list ``[]`` when no valid damage is found.

        Why a list?
        -----------
        A single part crop may genuinely contain multiple damage types at the
        same time (e.g. a dented door that also has a scratch).  Returning all
        confirmed types lets the registry accumulate independent vote tallies
        for each, producing a more complete final damage report.
        """
        if crop.size == 0:
            return []

        results = self.model.predict(crop, conf=conf_floor, verbose=False)
        boxes   = results[0].boxes
        masks   = results[0].masks

        # Collect the best confidence per allowed damage type
        best_per_type: Dict[str, Tuple[float, int]] = {}  # type → (conf, box_idx)
        for idx, box in enumerate(boxes):
            d_cls  = int(box.cls[0])
            d_type = self.model.names[d_cls]
            d_conf = float(box.conf[0])

            if d_type not in allowed_damages:
                continue
            if not passes_damage_threshold(d_type, d_conf):
                continue
            if d_type not in best_per_type or d_conf > best_per_type[d_type][0]:
                best_per_type[d_type] = (d_conf, idx)

        if not best_per_type:
            return []

        # Build the result list — one entry per distinct damage type, sorted by
        # descending confidence so the caller can easily identify the primary damage.
        dmg_list: List[Tuple[str, float, Optional[np.ndarray]]] = []
        for d_type, (d_conf, box_idx) in sorted(
            best_per_type.items(), key=lambda kv: kv[1][0], reverse=True
        ):
            mask_pts = None
            if masks is not None and box_idx < len(masks.xy):
                mask_pts = masks.xy[box_idx].astype(np.int32)
            dmg_list.append((d_type, d_conf, mask_pts))

        return dmg_list


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class CarDamagePipeline:
    """
    Orchestrates the full three-model pipeline for video inference.

    Frame-level flow (process_frame)
    ---------------------------------
    ┌─────────────────────────────────────────────────────────────────────┐
    │ a) DirectionClassifier.predict(frame)                               │
    │    → raw camera label + confidence                                  │
    │                                                                     │
    │ b) DirectionBuffer.update(car_direction, conf)                      │
    │    → stable_direction  (flicker-smoothed)                           │
    │                                                                     │
    │ c) CAR_PARTS_MAP[stable_direction]                                  │
    │    → allowed_parts list for this view                               │
    │                                                                     │
    │ d) PartsSegmenter.predict(frame, allowed_parts)                     │
    │    → list of filtered part detections                               │
    │                                                                     │
    │ e) For each detected part:                                          │
    │      • mark_part_seen in DamageRegistry                             │
    │      • DamageDetector.predict_on_crop(crop, allowed_damages)        │
    │      • If damage found → DamageRegistry.update(...)                 │
    │                                                                     │
    │ f) Return annotated frames (parts + damage) for display / writing   │
    └─────────────────────────────────────────────────────────────────────┘

    After all frames are processed, call DamageRegistry.finalize() to
    obtain the confirmed, vote-filtered damage report.
    """

    def __init__(
        self,
        parts_conf_floor:  float = 0.30,
        damage_conf_floor: float = 0.30,
        min_votes:         int   = REGISTRY_MIN_VOTES,
        min_ratio:         float = REGISTRY_MIN_VOTE_RATIO,
    ) -> None:
        log.info("Loading models …")
        self.direction_clf  = DirectionClassifier(MODEL_ANGLE_PATH)
        self.parts_seg      = PartsSegmenter(MODEL_PARTS_PATH)
        self.damage_det     = DamageDetector(MODEL_DAMAGE_PATH)
        log.info("Models loaded ✓")

        self.parts_conf_floor  = parts_conf_floor
        self.damage_conf_floor = damage_conf_floor

        self.dir_buffer = DirectionBuffer()
        self.registry   = DamageRegistry(min_votes=min_votes, min_ratio=min_ratio)

        # font config (used by drawing helpers)
        self._font      = cv2.FONT_HERSHEY_SIMPLEX
        self._fscale    = 0.42
        self._thickness = 1

    # ------------------------------------------------------------------
    # 8a. PUBLIC — process a single frame
    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame:       np.ndarray,
        frame_index: int,
        track_id:    int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """
        Run the full 5-step pipeline on one frame.

        Parameters
        ----------
        frame       : raw BGR frame from VideoCapture
        frame_index : global frame counter (used for registry records)
        track_id    : vehicle track ID (from ByteTrack); use 0 for single-car videos

        Returns
        -------
        parts_frame   : frame annotated with part segmentations
        damage_frame  : frame annotated with damage detections
        stable_dir    : current stable car-centric direction (or None)
        """
        parts_frame  = frame.copy()
        damage_frame = frame.copy()
        overlay_p    = parts_frame.copy()

        # ── Step a: Direction classification ─────────────────────────────────
        raw_label, car_direction, dir_conf = self.direction_clf.predict(frame)

        # ── Step b: Flicker suppression ───────────────────────────────────────
        # When the car is turning or the frame is motion-blurred, the classifier
        # may oscillate between adjacent directions (e.g., front ↔ front-left).
        # The DirectionBuffer absorbs these transients by voting over the last N
        # frames and returning the modal direction.
        stable_dir = self.dir_buffer.update(car_direction, dir_conf)

        if stable_dir is None:
            # Buffer still warming up — not enough frames yet to vote
            return parts_frame, damage_frame, None

        # ── Step c: Context-aware part list ───────────────────────────────────
        allowed_parts = CAR_PARTS_MAP.get(stable_dir, [])
        if not allowed_parts:
            # Unknown direction → skip this frame safely
            return parts_frame, damage_frame, stable_dir

        # ── Step d: Part segmentation (filtered) ──────────────────────────────
        detected_parts = self.parts_seg.predict(
            frame, allowed_parts, self.parts_conf_floor
        )

        # Deduplicate: keep only the highest-confidence detection per part name
        # so that mark_part_seen is called exactly once per part per frame and
        # the vote denominator stays accurate.
        best_per_part: Dict[str, Dict] = {}
        for p in detected_parts:
            name = p["part_name"]
            if name not in best_per_part or p["conf"] > best_per_part[name]["conf"]:
                best_per_part[name] = p
        detected_parts = list(best_per_part.values())

        drawings: List[Dict] = []

        for part_info in detected_parts:
            part_name  = part_info["part_name"]
            part_conf  = part_info["conf"]
            x1, y1, x2, y2 = part_info["bbox"]
            mask_pts   = part_info["mask_pts"]
            color      = part_color(part_name)

            # Mark that this part was visible (for ratio denominator).
            # stable_dir is passed so the counter is tracked per (part, direction)
            # key — keeping the vote denominator accurate for each directional view.
            self.registry.mark_part_seen(track_id, part_name, stable_dir)

            # Draw segmentation on parts frame
            if mask_pts is not None and mask_pts.size > 0:
                cv2.fillPoly(overlay_p, [mask_pts], color)
                cv2.polylines(parts_frame, [mask_pts], isClosed=True,
                              color=color, thickness=2)
            self._draw_label(
                parts_frame, f"{part_name} ({part_conf:.2f})",
                x1, y1, color
            )

            # ── Step e: Damage detection on crop ─────────────────────────────
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                # Degenerate bbox (zero width or height) — skip safely.
                # YOLO can rarely emit such boxes on very small detections.
                continue
            allowed_damages = get_allowed_damage(part_name)
            dmg_results = self.damage_det.predict_on_crop(
                crop, allowed_damages, self.damage_conf_floor
            )

            # predict_on_crop returns a list — one entry per distinct damage
            # type confirmed on this crop.  Iterate all of them so that a
            # part with both a dent AND a scratch is fully registered and drawn.
            # label_idx is used below to stack multiple HUD labels vertically
            # so they don't overlap when several damage types share one bbox.
            location = resolve_damage_location(part_name, stable_dir)
            for label_idx, (d_type, d_conf, d_mask_pts) in enumerate(dmg_results):

                # Translate crop-relative mask coordinates to frame coordinates
                if d_mask_pts is not None:
                    d_mask_pts = (d_mask_pts + np.array([x1, y1])).astype(np.int32)

                # Update the temporal registry (vote tallies are per damage_type,
                # so each type accumulates its own independent evidence)
                self.registry.update(
                    track_id=track_id,
                    part_name=part_name,
                    damage_type=d_type,
                    confidence=d_conf,
                    car_direction=stable_dir,
                    frame_index=frame_index,
                )

                drawings.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "part_name":  part_name,
                    "color":      color,
                    "d_mask":     d_mask_pts,
                    "d_type":     d_type,
                    "d_conf":     d_conf,
                    "location":   location,
                    "label_idx":  label_idx,
                })

        # Blend part segmentation masks
        if detected_parts:
            cv2.addWeighted(overlay_p, 0.30, parts_frame, 0.70, 0, parts_frame)

        # Annotate damage frame.
        # Multiple damage types on the same part share the same bbox anchor,
        # so we stack their labels upward using label_idx to avoid overlap.
        _LABEL_LINE_H = 18  # vertical step between stacked damage labels (px)
        for d in drawings:
            cv2.rectangle(damage_frame,
                          (d["x1"], d["y1"]), (d["x2"], d["y2"]),
                          d["color"], 2)
            if d["d_mask"] is not None:
                cv2.polylines(damage_frame, [d["d_mask"]],
                              isClosed=True, color=(0, 0, 255), thickness=2)
            label = f"{d['location']}: {d['d_type']} ({d['d_conf']:.2f})"
            # Shift each additional damage label one row higher so they
            # don't all render on top of each other.
            label_y = max(d["y1"] - d["label_idx"] * _LABEL_LINE_H, _LABEL_LINE_H)
            self._draw_label(damage_frame, label, d["x1"], label_y, d["color"])

        return parts_frame, damage_frame, stable_dir

    # ------------------------------------------------------------------
    # 8b. Drawing helper
    # ------------------------------------------------------------------
    def _draw_label(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ) -> None:
        (w, h), _ = cv2.getTextSize(text, self._font, self._fscale, self._thickness)
        label_y = y - 6 if y - h - 6 > 0 else y + h + 6
        cv2.rectangle(img, (x, label_y - h - 4), (x + w + 4, label_y + 4), color, -1)
        cv2.putText(img, text, (x + 2, label_y),
                    self._font, self._fscale, (255, 255, 255),
                    self._thickness, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  VIDEO RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total: int,
    fps: float,
    direction: Optional[str],
    mode: str,
) -> None:
    """Burn a compact status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    dir_txt   = direction or "warming up…"
    # Some container formats (e.g. certain H.264 streams) return 0 for
    # CAP_PROP_FRAME_COUNT.  Show "?" instead of a misleading "0".
    total_txt = str(total) if total > 0 else "?"
    txt = (
        f"Frame {frame_idx}/{total_txt}  "
        f"FPS:{fps:.1f}  "
        f"Dir:{dir_txt}  [{mode}]"
    )
    cv2.rectangle(frame, (0, h - 22), (w, h), (18, 18, 18), -1)
    cv2.putText(frame, txt, (8, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def run_video(
    video_path:        str,
    output_path:       str   = "result_pipeline.mp4",
    parts_conf:        float = 0.30,
    damage_conf:       float = 0.30,
    frame_skip:        int   = 1,
    preview:           bool  = False,
    save_parts:        bool  = True,
    save_damage:       bool  = True,
    report_path:       Optional[str] = None,
    min_votes:         int   = REGISTRY_MIN_VOTES,
    min_ratio:         float = REGISTRY_MIN_VOTE_RATIO,
    debug:             bool  = False,
) -> None:
    """
    Full end-to-end pipeline runner for a video file.

    Writes two annotated output videos (_parts, _damage) and optionally
    saves a JSON damage report with the finalized registry output.
    """
    log.info("=" * 65)
    log.info("Car Damage Detection — Integrated 3-Model Pipeline")
    log.info("Input  : %s", video_path)
    log.info("Output : %s", output_path)
    log.info("=" * 65)
    log.info(
        "Thresholds : votes>=%d  ratio>=%.0f%%  dir_conf>=%.2f  "
        "dir_buffer=%d",
        min_votes, min_ratio * 100, DIRECTION_CONF_THRESHOLD,
        DIRECTION_BUFFER_LEN,
    )

    pipeline = CarDamagePipeline(
        parts_conf_floor=parts_conf,
        damage_conf_floor=damage_conf,
        min_votes=min_votes,
        min_ratio=min_ratio,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps      = src_fps / frame_skip

    log.info("Resolution : %dx%d   FPS: %.1f → %.1f", width, height, src_fps, out_fps)
    log.info("Frames     : %d  (every %d)", total_frames, frame_skip)

    base   = os.path.splitext(output_path)[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer_parts: Optional[cv2.VideoWriter] = None
    if save_parts:
        writer_parts = cv2.VideoWriter(
            f"{base}_parts.mp4", fourcc, out_fps, (width, height)
        )
        if not writer_parts.isOpened():
            log.warning("VideoWriter for parts could not be opened — parts video will be skipped.")
            writer_parts = None

    writer_damage: Optional[cv2.VideoWriter] = None
    if save_damage:
        writer_damage = cv2.VideoWriter(
            f"{base}_damage.mp4", fourcc, out_fps, (width, height)
        )
        if not writer_damage.isOpened():
            log.warning("VideoWriter for damage could not be opened — damage video will be skipped.")
            writer_damage = None

    frame_idx = written = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if (frame_idx - 1) % frame_skip != 0:
                continue

            elapsed = time.time() - t0
            live_fps = written / elapsed if elapsed > 0 else 0.0
            print(f"  Frame {frame_idx:>5}/{total_frames}  live={live_fps:.1f} fps", end="\r")

            # ── Core pipeline call ────────────────────────────────────────────
            try:
                parts_frm, damage_frm, stable_dir = pipeline.process_frame(
                    frame, frame_index=frame_idx, track_id=0
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Frame %d skipped due to error: %s", frame_idx, exc)
                continue

            _draw_hud(parts_frm,  frame_idx, total_frames, live_fps, stable_dir, "PARTS")
            _draw_hud(damage_frm, frame_idx, total_frames, live_fps, stable_dir, "DAMAGE")

            if writer_parts  is not None:
                writer_parts.write(parts_frm)
            if writer_damage is not None:
                writer_damage.write(damage_frm)

            written += 1

            if preview:
                cv2.imshow("Parts",  parts_frm)
                cv2.imshow("Damage", damage_frm)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n  [Preview] Stopped early.")
                    break

    finally:
        cap.release()
        if writer_parts  is not None: writer_parts.release()
        if writer_damage is not None: writer_damage.release()
        if preview: cv2.destroyAllWindows()

    # ── Damage Report ─────────────────────────────────────────────────────────
    log.info("Processed %d frames in %.1fs", written, time.time() - t0)

    if debug:
        print(pipeline.registry.debug_registry())

    # Call finalize() exactly once so that PartRecord.confirmed_damages is
    # written only once.  Both the console summary and the optional JSON file
    # share the same pre-computed result — no redundant re-voting.
    report = pipeline.registry.finalize()
    print(DamageRegistry.format_report(report))

    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
        log.info("Report saved → %s", report_path)

    if writer_parts is not None:
        log.info("Parts video  → %s_parts.mp4", base)
    if writer_damage is not None:
        log.info("Damage video → %s_damage.mp4", base)
    log.info("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  IMAGE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_image(
    image_path:        str,
    output_path:       str   = "result_pipeline.jpg",
    parts_conf:        float = 0.30,
    damage_conf:       float = 0.30,
    report_path:       Optional[str] = None,
    debug:             bool  = False,
) -> None:
    """
    Single-image inference mode.

    Key differences from run_video
    -------------------------------
    • No VideoCapture loop — the image is treated as a single frame.
    • DirectionBuffer still runs but receives exactly one observation;
      because maxlen=5 and the buffer is pre-warmed after the first
      high-conf update, stable_dir is set immediately.
    • DamageRegistry is created with min_votes=1 and min_ratio=0.0 so
      that a single-frame detection counts as "confirmed" — temporal
      voting only makes sense across multiple frames.
    • Output is two annotated images (_parts / _damage) instead of videos.
    """
    log.info("=" * 65)
    log.info("Car Damage Detection — Image Mode")
    log.info("Input  : %s", image_path)
    log.info("Output : %s", output_path)
    log.info("=" * 65)

    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(
            f"Cannot load image: {image_path}\n"
            "Check the path is correct and the file is a supported format "
            f"({', '.join(sorted(IMAGE_EXTENSIONS))})."
        )

    h, w = frame.shape[:2]
    log.info("Resolution : %dx%d", w, h)

    # For a single image bypass temporal voting — every detection is
    # immediately confirmed (min_votes=1, min_ratio=0.0).
    pipeline = CarDamagePipeline(
        parts_conf_floor=parts_conf,
        damage_conf_floor=damage_conf,
        min_votes=1,
        min_ratio=0.0,
    )

    parts_frm, damage_frm, stable_dir = pipeline.process_frame(
        frame, frame_index=1, track_id=0
    )

    if stable_dir is None:
        log.warning(
            "Direction classifier returned no stable direction on this image.\n"
            "The output frames may be unannotated.  Try a clearer, well-lit photo."
        )
    else:
        log.info("Detected direction : %s", stable_dir)

    if debug:
        print(pipeline.registry.debug_registry())

    report = pipeline.registry.finalize()
    print(DamageRegistry.format_report(report))

    # ── Save annotated output images ──────────────────────────────────────────
    base = os.path.splitext(output_path)[0]
    ext  = os.path.splitext(output_path)[1] or ".jpg"

    parts_out  = f"{base}_parts{ext}"
    damage_out = f"{base}_damage{ext}"

    cv2.imwrite(parts_out,  parts_frm)
    cv2.imwrite(damage_out, damage_frm)
    log.info("Parts image  → %s", parts_out)
    log.info("Damage image → %s", damage_out)

    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
        log.info("Report saved → %s", report_path)

    log.info("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Integrated 3-Model Car Damage Pipeline (video & image)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples — Video:
  python pipeline.py testvideo2.mp4
  python pipeline.py testvideo2.mp4 --frame-skip 2 --preview
  python pipeline.py testvideo2.mp4 --report report.json
  python pipeline.py testvideo2.mp4 --min-votes 5 --min-ratio 0.4

Examples — Image:
  python pipeline.py car.jpg
  python pipeline.py car.png --output result.jpg --report report.json
  python pipeline.py car.jpg --parts-conf 0.25 --damage-conf 0.25 --debug
        """,
    )
    ap.add_argument("input",
                    help="Input file — video (mp4/avi/…) or image (jpg/png/bmp/…).\n"
                         "Mode is auto-detected from the file extension.")
    ap.add_argument("--output",      default="result_pipeline",
                    help="Base output path without extension (default: result_pipeline).\n"
                         "Extensions are added automatically (_parts.mp4 / _parts.jpg etc.).")
    ap.add_argument("--parts-conf",  type=float, default=0.30,
                    help="Parts segmentation conf floor (default: 0.30)")
    ap.add_argument("--damage-conf", type=float, default=0.30,
                    help="Damage detection conf floor (default: 0.30)")
    # ── Video-only flags ───────────────────────────────────────────────────────
    ap.add_argument("--frame-skip",  type=int, default=1,
                    help="[Video only] Process every Nth frame (default: 1)")
    ap.add_argument("--preview",     action="store_true",
                    help="[Video only] Show live preview windows (press q to stop)")
    ap.add_argument("--no-parts",    action="store_true",
                    help="[Video only] Skip parts output video")
    ap.add_argument("--no-damage",   action="store_true",
                    help="[Video only] Skip damage output video")
    ap.add_argument("--min-votes",   type=int, default=REGISTRY_MIN_VOTES,
                    help=f"[Video only] Min frames to confirm damage (default: {REGISTRY_MIN_VOTES})")
    ap.add_argument("--min-ratio",   type=float, default=REGISTRY_MIN_VOTE_RATIO,
                    help=f"[Video only] Min frame ratio to confirm damage (default: {REGISTRY_MIN_VOTE_RATIO})")
    # ── Shared flags ───────────────────────────────────────────────────────────
    ap.add_argument("--report",      default=None,
                    help="Optional path to save JSON damage report")
    ap.add_argument("--debug",       action="store_true",
                    help="Print raw vote counts from registry before final report")
    args = ap.parse_args()

    _ext = Path(args.input).suffix.lower()
    if _ext in IMAGE_EXTENSIONS:
        # ── Image mode ────────────────────────────────────────────────────────
        # Default output keeps the same extension as the input image.
        _out = args.output
        if not Path(_out).suffix:
            _out = _out + _ext          # e.g. "result_pipeline" → "result_pipeline.jpg"
        run_image(
            image_path  = args.input,
            output_path = _out,
            parts_conf  = args.parts_conf,
            damage_conf = args.damage_conf,
            report_path = args.report,
            debug       = args.debug,
        )
    else:
        # ── Video mode ────────────────────────────────────────────────────────
        _out = args.output
        if not Path(_out).suffix:
            _out = _out + ".mp4"        # e.g. "result_pipeline" → "result_pipeline.mp4"
        run_video(
            video_path  = args.input,
            output_path = _out,
            parts_conf  = args.parts_conf,
            damage_conf = args.damage_conf,
            frame_skip  = args.frame_skip,
            preview     = args.preview,
            save_parts  = not args.no_parts,
            save_damage = not args.no_damage,
            report_path = args.report,
            min_votes   = args.min_votes,
            min_ratio   = args.min_ratio,
            debug       = args.debug,
        )
