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
from typing import Dict, List, Optional, Set, Tuple

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

BASE_DIR = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION — edit thresholds here without touching pipeline logic
# ══════════════════════════════════════════════════════════════════════════════

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_ANGLE_PATH  = BASE_DIR / "models" / "best_car_angle.pt"
MODEL_PARTS_PATH  = BASE_DIR / "models" / "best_car_part.pt"
MODEL_DAMAGE_PATH = BASE_DIR / "models" / "best_damage_type.pt"

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
        "Front-bumper", "Fender", "Mirror",
        "Hood", "Headlight", "Windshield",
    ],
    "front-right-side": [
        "Front-bumper", "Fender", "Mirror",
        "Hood", "Headlight", "Windshield",
    ],
    "back-left-side": [
        "Back-bumper", "Quarter-panel",
        "Tail-light", "Back-windshield",
    ],
    "back-right-side": [
        "Back-bumper", "Quarter-panel",
        "Tail-light", "Back-windshield",
    ],
}

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
    "Hood":            ["dent", "scratch", "crack"],
    "Trunk":           ["dent", "scratch", "crack"],
    "Roof":            ["dent", "scratch", "crack"],
    "Fender":          ["dent", "scratch", "crack"],
    "Front-door":      ["dent", "scratch", "crack"],
    "Back-door":       ["dent", "scratch", "crack"],
    "Quarter-panel":   ["dent", "scratch", "crack"],
    "Rocker-panel":    ["dent", "scratch", "crack"],
}

# ── DamageRegistry voting parameters ─────────────────────────────────────────
# All temporal thresholds are expressed in SECONDS — FPS-agnostic by design.
# compute_adaptive_thresholds() converts them to frame counts at runtime.
REGISTRY_MIN_VOTE_SECONDS = 0.20  # damage must be visible for >= 0.20 s to be confirmed
REGISTRY_MIN_VOTE_RATIO   = 0.15  # damage seen in >= 15 % of frames it was observable
INSTANCE_MATCH_RADIUS     = 0.30


def compute_adaptive_thresholds(fps: float) -> dict:
    """
    Convert time-based threshold constants into frame counts for a given FPS.

    This makes the temporal voting logic FPS-agnostic:
      • At 15 FPS  →  min_votes = 3
      • At 30 FPS  →  min_votes = 6
      • At 60 FPS  →  min_votes = 12

    Parameters
    ----------
    fps : Effective frames-per-second of the processed stream (src_fps / frame_skip).
          Must be > 0; raises ValueError otherwise.

    Returns
    -------
    dict with keys:
        min_votes        – minimum frame-vote count to confirm a damage
        min_ratio        – minimum vote / frames-seen ratio (unchanged)
    """
    if fps <= 0:
        raise ValueError(
            f"compute_adaptive_thresholds: fps must be > 0, got {fps}. "
            "Check that the video file reports a valid frame rate."
        )

    min_votes = max(1, round(fps * REGISTRY_MIN_VOTE_SECONDS))

    log.info(
        "Adaptive thresholds @ %.1f FPS → min_votes=%d (%.2fs)  min_ratio=%.0f%%",
        fps, min_votes, REGISTRY_MIN_VOTE_SECONDS, REGISTRY_MIN_VOTE_RATIO * 100
    )
    return {
        "min_votes": min_votes,
        "min_ratio": REGISTRY_MIN_VOTE_RATIO,
    }

# ── Severity classification thresholds ────────────────────────────────────────
# severity_ratio = damage_mask_area / part_mask_area
# Thresholds are checked in order; first match wins.
SEVERITY_THRESHOLDS: List[Tuple[float, str]] = [
    (0.05, "Minor"),     # 0 %  –  5 %
    (0.20, "Moderate"),  # 5 %  – 20 %
    (1.00, "Severe"),    # 20 % +
]


def classify_severity(ratio: float) -> str:
    """Map a damage-area / part-area ratio to a human-readable severity label."""
    for threshold, label in SEVERITY_THRESHOLDS:
        if ratio <= threshold:
            return label
    return "Severe"  # anything above 100 % (shouldn't happen, but safe fallback)


# ── Side parts and resolution helper ──────────────────────────────────────────
# Prepend "Left" or "Right" to side-specific parts based on the camera view
# direction so that side parts are distinguished by left/right side, while
# bumpers and other non-side parts are grouped globally.
SIDE_PARTS: Set[str] = {
    "Front-door", "Back-door", "Front-wheel", "Back-wheel",
    "Front-window", "Back-window", "Fender", "Quarter-panel",
    "Mirror", "Rocker-panel", "Headlight", "Tail-light"
}

def resolve_side_part_name(part_name: str, car_direction: str) -> str:
    """Prepend 'Left ' or 'Right ' to side parts based on camera direction.
    Bumpers, hood, windshield, etc. are left unchanged so they cluster globally.
    """
    if part_name in SIDE_PARTS:
        if "left" in car_direction.lower():
            return f"Left {part_name}"
        elif "right" in car_direction.lower():
            return f"Right {part_name}"
    return part_name


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

def ensure_parent_dir(path: str | Path) -> None:
    """Create an output file's parent directory when it is explicit."""
    parent = Path(path).expanduser().parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def validate_confidence(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


def validate_runtime_options(
    parts_conf: float,
    damage_conf: float,
) -> None:
    """Validate only the model confidence floors — voting thresholds are adaptive."""
    validate_confidence("parts_conf", parts_conf)
    validate_confidence("damage_conf", damage_conf)


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
    return PART_DAMAGE_MAP.get(part_name)


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
class DamageInstance:
    """
    Accumulated damage evidence for one damage type on a part.

    Fields
    ------
    damage_type    : e.g. "scratch"
    vote_count     : number of frames this instance was observed
    best_conf      : highest confidence seen across all observations
    cx_norm, cy_norm : running mean of the normalised centroid
    frames_seen    : set of frame indices (prevents double-counting)
    location       : human-readable compound label (most-voted)
    votes_per_direction: map from camera direction string to vote count
    best_damage_area_px : largest damage mask pixel area seen (for severity)
    best_part_area_px   : part mask pixel area from the frame where
                          best_damage_area_px was recorded
    """
    damage_type: str
    vote_count:  int = 0
    best_conf:   float = 0.0
    cx_norm:     float = 0.0
    cy_norm:     float = 0.0
    frames_seen: Set[int] = field(default_factory=set)
    _loc_votes:  Dict[str, int] = field(default_factory=dict)
    votes_per_direction: Dict[str, int] = field(default_factory=dict)
    best_damage_area_px: float = 0.0
    best_part_area_px:   float = 0.0

    def update(
        self,
        conf: float,
        cx: float,
        cy: float,
        frame_index: int,
        location: str,
        direction: str,
        damage_area_px: float = 0.0,
        part_area_px: float = 0.0,
    ) -> None:
        """Absorb a new observation into this instance."""
        if frame_index in self.frames_seen:
            # Already counted this frame — only update confidence if higher
            if conf > self.best_conf:
                self.best_conf = conf
            return
        self.frames_seen.add(frame_index)
        self.vote_count += 1
        if conf > self.best_conf:
            self.best_conf = conf
        # Incremental mean update for the centroid
        n = self.vote_count
        self.cx_norm += (cx - self.cx_norm) / n
        self.cy_norm += (cy - self.cy_norm) / n
        self._loc_votes[location] = self._loc_votes.get(location, 0) + 1
        self.votes_per_direction[direction] = self.votes_per_direction.get(direction, 0) + 1
        # Track the largest observed damage area and largest observed part
        # area INDEPENDENTLY.  Using max(damage) / max(part) keeps the ratio
        # bounded (≤ 1.0) because the largest part view is always at least as
        # large as the largest damage view on that same part.
        if damage_area_px > self.best_damage_area_px:
            self.best_damage_area_px = damage_area_px
        if part_area_px > self.best_part_area_px:
            self.best_part_area_px = part_area_px

    @property
    def severity_ratio(self) -> float:
        """Ratio of damage area to part area, clamped to [0.0, 1.0]."""
        if self.best_part_area_px > 0:
            return min(self.best_damage_area_px / self.best_part_area_px, 1.0)
        return 0.0

    @property
    def severity(self) -> str:
        """Human-readable severity label derived from severity_ratio."""
        return classify_severity(self.severity_ratio)

    @property
    def best_location(self) -> str:
        """Most-voted human-readable location label for this instance."""
        if not self._loc_votes:
            return ""
        return max(self._loc_votes, key=self._loc_votes.__getitem__)


@dataclass
class PartRecord:
    """
    Aggregated record for a single (track_id, part_name) pair.

    Fields
    ------
    instances     : damage_type → list of spatial clusters (DamageInstance)
    _seen_frames_per_dir : map from camera direction string to set of unique frame indices where this part was visible
    """
    part_name:     str
    track_id:      int
    # damage_type → list of DamageInstance clusters
    instances:    Dict[str, List[DamageInstance]] = field(default_factory=dict)
    _seen_frames_per_dir: Dict[str, Set[int]] = field(default_factory=dict)
    # Max part mask area observed across ALL frames and ALL camera angles.
    # Used as the severity denominator so the ratio reflects the most
    # complete view of the part, not just the angle where damage was found.
    max_part_area_px: float = 0.0

    @property
    def total_frames_seen(self) -> int:
        """Number of unique frames in which this part was detected."""
        union_frames = set()
        for f_set in self._seen_frames_per_dir.values():
            union_frames.update(f_set)
        return len(union_frames)

    def mark_seen(self, frame_index: int, direction: str, part_area_px: float = 0.0) -> None:
        """Record that this part was visible on a given frame under a specific direction."""
        if direction not in self._seen_frames_per_dir:
            self._seen_frames_per_dir[direction] = set()
        self._seen_frames_per_dir[direction].add(frame_index)
        if part_area_px > self.max_part_area_px:
            self.max_part_area_px = part_area_px

    def add_damage_observation(
        self,
        damage_type: str,
        confidence: float,
        frame_index: int,
        location: str,
        direction: str,
        damage_mask: Optional[np.ndarray],
        crop_size: Tuple[int, int],
        damage_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0),
        match_radius: float = INSTANCE_MATCH_RADIUS,
        part_area_px: float = 0.0,
    ) -> None:
        """
        Record a damage observation of a given type, clustering spatially.

        Geometry is derived from the segmentation mask when available:
          • Centroid is computed via cv2.moments on the crop-relative mask polygon.
          • Falls back to the bounding-box centroid when damage_mask is None or
            degenerate (e.g. fewer than 3 points).
        """
        if direction not in self._seen_frames_per_dir:
            self._seen_frames_per_dir[direction] = set()
        self._seen_frames_per_dir[direction].add(frame_index)
        if part_area_px > self.max_part_area_px:
            self.max_part_area_px = part_area_px

        crop_w, crop_h = crop_size
        cx, cy = 0.5, 0.5  # safe defaults

        # ── Compute damage mask area (pixels) for severity ────────────────
        damage_area_px = 0.0
        if damage_mask is not None and damage_mask.size >= 6:
            damage_area_px = float(cv2.contourArea(damage_mask.astype(np.float32)))

        if damage_mask is not None and damage_mask.size >= 6:
            # Compute the true geometric centroid of the segmentation mask using
            # OpenCV image moments.  The mask polygon is in crop-relative coords.
            M = cv2.moments(damage_mask.astype(np.float32))
            if M["m00"] > 1e-6:
                # Normalise by the crop dimensions, identical normalisation to
                # the old box-centroid path so the match_radius threshold stays valid.
                cx = (M["m10"] / M["m00"]) / crop_w if crop_w > 0 else 0.5
                cy = (M["m01"] / M["m00"]) / crop_h if crop_h > 0 else 0.5
            elif crop_w > 0 and crop_h > 0:
                # Degenerate moment (zero-area contour) — fall back to bbox centroid
                bx1, by1, bx2, by2 = damage_bbox
                cx = ((bx1 + bx2) / 2.0) / crop_w
                cy = ((by1 + by2) / 2.0) / crop_h
        elif crop_w > 0 and crop_h > 0:
            # No mask available — fall back to bounding-box centroid
            bx1, by1, bx2, by2 = damage_bbox
            cx = ((bx1 + bx2) / 2.0) / crop_w
            cy = ((by1 + by2) / 2.0) / crop_h

        # When no mask is available, approximate damage area from bbox
        if damage_area_px == 0.0:
            bx1, by1, bx2, by2 = damage_bbox
            damage_area_px = float(max(0, bx2 - bx1) * max(0, by2 - by1))

        if damage_type not in self.instances:
            self.instances[damage_type] = []

        # Find the closest existing instance of this damage type
        existing_list = self.instances[damage_type]
        best_inst: Optional[DamageInstance] = None
        best_dist: float = float("inf")

        for inst in existing_list:
            # Aspect-Ratio Preserving Distance
            w2 = crop_w ** 2
            h2 = crop_h ** 2
            diag2 = w2 + h2
            if diag2 > 0:
                w_weight = w2 / diag2
                h_weight = h2 / diag2
            else:
                w_weight, h_weight = 0.5, 0.5
            
            dist = (w_weight * (cx - inst.cx_norm) ** 2 + 
                    h_weight * (cy - inst.cy_norm) ** 2) ** 0.5
            
            if dist < best_dist:
                best_dist = dist
                best_inst = inst

        if best_inst is not None and best_dist <= match_radius:
            best_inst.update(confidence, cx, cy, frame_index, location, direction,
                            damage_area_px=damage_area_px, part_area_px=part_area_px)
        else:
            new_inst = DamageInstance(damage_type=damage_type)
            new_inst.update(confidence, cx, cy, frame_index, location, direction,
                           damage_area_px=damage_area_px, part_area_px=part_area_px)
            existing_list.append(new_inst)

    def confirmed_instances(
        self,
        min_votes: int   = 3,
        min_ratio: float = 0.15,
    ) -> List[DamageInstance]:
        """
        Return all damage instances that pass the vote thresholds.

        An instance is confirmed if in at least one camera direction:
          1. vote_count in direction >= min_votes
          2. vote_count in direction / total frames seen in that direction >= min_ratio

        Before returning, each confirmed instance's best_part_area_px is
        upgraded to the PartRecord's max_part_area_px (the largest part
        area observed across ALL frames and ALL camera angles).  This
        ensures the severity denominator reflects the most complete view
        of the part, not just the angle where damage happened to be detected.
        """
        result: List[DamageInstance] = []
        for inst_list in self.instances.values():
            for inst in inst_list:
                confirmed = False
                for direction, votes in inst.votes_per_direction.items():
                    seen_in_dir = len(self._seen_frames_per_dir.get(direction, set()))
                    if seen_in_dir > 0:
                        ratio = votes / seen_in_dir
                        if votes >= min_votes and ratio >= min_ratio:
                            confirmed = True
                            break
                if confirmed:
                    # Use the max part area across all angles for severity
                    if self.max_part_area_px > inst.best_part_area_px:
                        inst.best_part_area_px = self.max_part_area_px
                    result.append(inst)
        return result


class DamageRegistry:
    """
    Central store that accumulates damage evidence across all video frames.
    Tracks damage instances keyed on (track_id, resolved_part_name) to distinguish left/right side parts
    while allowing bumpers to cluster globally across all camera views.
    """

    def __init__(
        self,
        min_votes: int   = 3,
        min_ratio: float = 0.15,
    ) -> None:
        self._min_votes = min_votes
        self._min_ratio = min_ratio
        # key: (track_id, resolved_part_name)  →  PartRecord
        self._records: Dict[Tuple[int, str], PartRecord] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        track_id:     int,
        part_name:    str,
        damage_type:  str,
        confidence:   float,
        car_direction: str,
        frame_index:  int,
        damage_mask:  Optional[np.ndarray] = None,
        damage_bbox:  Tuple[int, int, int, int] = (0, 0, 0, 0),
        crop_size:    Tuple[int, int] = (0, 0),
        location:     Optional[str] = None,
        part_area_px: float = 0.0,
    ) -> None:
        """Record a single damage observation from one frame.

        Parameters
        ----------
        damage_mask : crop-relative segmentation mask polygon (int32 ndarray).
                      Used to compute the mask centroid for instance matching.
                      When None, falls back to the bbox centroid (legacy path).
        damage_bbox : crop-relative bounding box — kept as fallback when the
                      mask is unavailable.
        crop_size   : (width, height) of the part crop
        part_area_px : pixel area of the part's segmentation mask (for severity)
        """
        resolved_part = resolve_side_part_name(part_name, car_direction)
        key = (track_id, resolved_part)
        if key not in self._records:
            self._records[key] = PartRecord(
                part_name=resolved_part, track_id=track_id
            )

        display_location = location or resolved_part.replace("-", " ").title()
        self._records[key].add_damage_observation(
            damage_type=damage_type,
            confidence=confidence,
            frame_index=frame_index,
            location=display_location,
            direction=car_direction,
            damage_mask=damage_mask,
            crop_size=crop_size,
            damage_bbox=damage_bbox,
            part_area_px=part_area_px,
        )

    def mark_part_seen(
        self,
        track_id:      int,
        part_name:     str,
        car_direction: str,
        frame_index:   int,
        part_area_px:  float = 0.0,
    ) -> None:
        """
        Track that a part was visible on a given frame, even when no damage
        is detected on that frame.  Also tracks the max part area across
        all frames for accurate severity computation.
        """
        resolved_part = resolve_side_part_name(part_name, car_direction)
        key = (track_id, resolved_part)
        if key not in self._records:
            self._records[key] = PartRecord(
                part_name=resolved_part, track_id=track_id
            )
        self._records[key].mark_seen(frame_index, car_direction, part_area_px=part_area_px)

    # ------------------------------------------------------------------
    def finalize(self) -> List[Dict]:
        """
        Run voting logic on all records and return the final damage report.
        """
        report: List[Dict] = []

        for (track_id, part_name), record in sorted(self._records.items()):
            for inst in record.confirmed_instances(self._min_votes, self._min_ratio):
                report.append({
                    "track_id":     track_id,
                    "part_name":    part_name,
                    "damage_type":  inst.damage_type,
                    "confidence":   round(inst.best_conf, 4),
                    "severity":     inst.severity,
                    "severity_ratio": round(inst.severity_ratio, 4),
                })

        return report

    @staticmethod
    def format_report(report: List[Dict]) -> str:
        """
        Format a pre-finalized damage report list as a human-readable string.
        """
        if not report:
            return "No confirmed damage detected."
        lines = ["=" * 76, "  DAMAGE REPORT", "=" * 76]
        for item in report:
            sev = item.get('severity', 'N/A')
            sev_ratio = item.get('severity_ratio', 0.0)
            lines.append(
                f"  {item['part_name']:<30}  "
                f"{item['damage_type']:<14}  "
                f"conf={item['confidence']:.2f}  "
                f"severity={sev:<8} ({sev_ratio:.1%})"
            )
        lines.append("=" * 76)
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable summary of the finalized report."""
        return self.format_report(self.finalize())

    def debug_registry(self) -> str:
        """
        Print raw vote counts for EVERY detected part BEFORE the voting thresholds are applied.
        """
        if not self._records:
            return "\n".join([
                "=" * 80,
                "  DEBUG — Raw Registry State (before voting)",
                "=" * 80,
                "  Registry is empty — no parts were detected at all.",
                "=" * 80
            ])

        # Define columns and borders for the table format
        col_headers = [
            f"{'Track':<5}",
            f"{'Part Name':<22}",
            f"{'Damage Type':<12}",
            f"{'Votes':<5}",
            f"{'Seen':<5}",
            f"{'Centroid':<14}",
            f"{'Severity':<18}",
            f"{'Per-Direction Ratios (Votes/Seen)':<45}",
            f"{'Verdict':<9}"
        ]
        
        border_parts = [
            "-" * 5,
            "-" * 22,
            "-" * 12,
            "-" * 5,
            "-" * 5,
            "-" * 14,
            "-" * 18,
            "-" * 45,
            "-" * 9
        ]
        
        header = "  " + " | ".join(col_headers)
        separator = "  " + "-+-".join(border_parts)
        
        lines = [
            "=" * 155,
            "  DEBUG — RAW REGISTRY STATE (BEFORE VOTING)",
            "=" * 155,
            header,
            separator
        ]

        for (track_id, part_name), record in sorted(self._records.items()):
            # Collect all instances across damage types for debug display
            all_instances: List[DamageInstance] = [
                inst
                for inst_list in record.instances.values()
                for inst in inst_list
            ]

            total_part_seen = record.total_frames_seen

            if not all_instances:
                seen_dirs = [f"{d}:{len(f)}" for d, f in sorted(record._seen_frames_per_dir.items())]
                seen_txt = f"No damage; seen in: {', '.join(seen_dirs)}"
                lines.append(
                    f"  {track_id:<5} | {part_name:<22} | {'-':<12} | {'-':<5} | {total_part_seen:<5} | {'-':<14} | {'-':<18} | {seen_txt:<45} | {'-':<9}"
                )
            else:
                # Sort by damage type then descending vote count
                for inst in sorted(all_instances,
                                   key=lambda i: (i.damage_type, -i.vote_count)):
                    dir_strings = []
                    confirmed = False
                    for direction, votes in sorted(inst.votes_per_direction.items()):
                        seen_in_dir = len(record._seen_frames_per_dir.get(direction, set()))
                        ratio_pct = (votes / seen_in_dir * 100.0) if seen_in_dir > 0 else 0.0
                        
                        passes_v = votes >= self._min_votes
                        passes_r = (votes / seen_in_dir) >= self._min_ratio if seen_in_dir > 0 else False
                        if passes_v and passes_r:
                            confirmed = True
                        
                        dir_strings.append(f"{direction}:{votes}/{seen_in_dir} ({ratio_pct:.0f}%)")
                    
                    dir_txt = ", ".join(dir_strings)
                    centroid_txt = f"({inst.cx_norm:.2f}, {inst.cy_norm:.2f})"
                    verdict = "CONFIRMED" if confirmed else "FAIL"
                    sev_txt = f"{inst.severity} ({inst.severity_ratio:.1%})"
                    
                    lines.append(
                        f"  {track_id:<5} | {part_name:<22} | {inst.damage_type:<12} | {inst.vote_count:<5} | {total_part_seen:<5} | {centroid_txt:<14} | {sev_txt:<18} | {dir_txt:<45} | {verdict:<9}"
                    )

        lines.append("=" * 155)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

class DirectionClassifier:
    """Wraps the YOLOv8 classification model for camera-angle prediction."""

    def __init__(self, model_path: str | Path = MODEL_ANGLE_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Direction model not found: {model_path}")
        self.model = YOLO(str(model_path))
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

    def __init__(self, model_path: str | Path = MODEL_PARTS_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Parts model not found: {model_path}")
        self.model = YOLO(str(model_path))

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

    def __init__(self, model_path: str | Path = MODEL_DAMAGE_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Damage model not found: {model_path}")
        self.model = YOLO(str(model_path))

    def predict_on_crop(
        self,
        crop: np.ndarray,
        allowed_damages: List[str],
        conf_floor: float = 0.30,
    ) -> List[Tuple[str, float, Optional[np.ndarray], Tuple[int, int, int, int]]]:
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

        # Collect ALL valid detections, keeping each box as a candidate instance.
        # Two detections of the same type that are spatially separated on the crop
        # are kept as distinct candidates; the registry will cluster them later.
        # Each candidate carries its crop-relative mask polygon so the NMS loop
        # can compare masks directly (mask IoU) rather than bounding boxes.
        candidates: List[Tuple[str, float, int, Tuple[int,int,int,int], Optional[np.ndarray]]] = []
        #            (damage_type, conf, box_idx, crop_bbox, mask_pts_crop_relative)
        for idx, box in enumerate(boxes):
            d_cls  = int(box.cls[0])
            d_type = self.model.names[d_cls]
            d_conf = float(box.conf[0])

            if d_type not in allowed_damages:
                continue

            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            # Extract crop-relative mask polygon eagerly so the NMS loop has it.
            cand_mask: Optional[np.ndarray] = None
            if masks is not None and idx < len(masks.xy):
                cand_mask = masks.xy[idx].astype(np.int32)
            candidates.append((d_type, d_conf, idx, (bx1, by1, bx2, by2), cand_mask))

        if not candidates:
            return []

        # ── Mask IoU helper ───────────────────────────────────────────────────
        # Merge duplicate detections of the *same* physical damage using pixel-
        # level mask IoU instead of axis-aligned box IoU.  Two candidates are
        # considered duplicates when their mask overlap exceeds 50 %.
        # Distinct spatial damages (low IoU) are preserved as separate entries.
        # Falls back to box IoU when either candidate has no valid mask.
        def _box_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
            """Axis-aligned bounding-box IoU (fallback when masks are absent)."""
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                return 0.0
            area_a = (a[2]-a[0]) * (a[3]-a[1])
            area_b = (b[2]-b[0]) * (b[3]-b[1])
            return inter / (area_a + area_b - inter)

        def _mask_iou(
            pts_a: Optional[np.ndarray],
            pts_b: Optional[np.ndarray],
            box_a: Tuple[int,int,int,int],
            box_b: Tuple[int,int,int,int],
            crop_shape: Tuple[int, int],
        ) -> float:
            """
            Pixel-level mask IoU computed by rendering both polygon contours
            into binary images and measuring intersection / union.

            Falls back to box IoU when either mask is missing or degenerate.

            Parameters
            ----------
            pts_a, pts_b : crop-relative contour point arrays (int32)
            box_a, box_b : crop-relative bounding boxes (for the fallback)
            crop_shape   : (height, width) of the crop image
            """
            if (
                pts_a is None or pts_b is None
                or pts_a.size < 6 or pts_b.size < 6
            ):
                # Not enough points to render a polygon — fall back to box IoU
                return _box_iou(box_a, box_b)

            h, w = crop_shape
            if h <= 0 or w <= 0:
                return _box_iou(box_a, box_b)

            # Render both polygons into binary masks of the same size
            mask_a = np.zeros((h, w), dtype=np.uint8)
            mask_b = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_a, [pts_a], 1)
            cv2.fillPoly(mask_b, [pts_b], 1)

            inter = int(np.logical_and(mask_a, mask_b).sum())
            union = int(np.logical_or(mask_a, mask_b).sum())
            return inter / union if union > 0 else 0.0

        # Sort by descending confidence so we always keep the best detection when merging
        candidates.sort(key=lambda c: c[1], reverse=True)
        kept_candidates: List[Tuple[str, float, int, Tuple[int,int,int,int], Optional[np.ndarray]]] = []
        crop_h_nms, crop_w_nms = crop.shape[:2]
        for cand in candidates:
            d_type, d_conf, box_idx, cbox, cmask = cand
            suppress = False
            for kept in kept_candidates:
                if kept[0] == d_type and _mask_iou(
                    cmask, kept[4], cbox, kept[3], (crop_h_nms, crop_w_nms)
                ) > 0.50:
                    suppress = True   # duplicate mask — already have a better one
                    break
            if not suppress:
                kept_candidates.append(cand)

        # Build the result list — one entry per surviving candidate instance
        dmg_list: List[Tuple[str, float, Optional[np.ndarray], Tuple[int,int,int,int]]] = []
        for d_type, d_conf, box_idx, cbox, cmask in kept_candidates:
            dmg_list.append((d_type, d_conf, cmask, cbox))

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
        fps:               float = 25.0,
    ) -> None:
        """
        Parameters
        ----------
        parts_conf_floor  : Minimum model confidence to accept a part detection.
        damage_conf_floor : Minimum model confidence to accept a damage detection.
        fps               : Effective frames-per-second of the processed stream
                            (src_fps / frame_skip).  All temporal thresholds
                            (min_votes, direction_streak) are computed from this
                            value via compute_adaptive_thresholds().
        """
        log.info("Loading models …")
        self.direction_clf  = DirectionClassifier(MODEL_ANGLE_PATH)
        self.parts_seg      = PartsSegmenter(MODEL_PARTS_PATH)
        self.damage_det     = DamageDetector(MODEL_DAMAGE_PATH)
        log.info("Models loaded ✓")

        self.parts_conf_floor  = parts_conf_floor
        self.damage_conf_floor = damage_conf_floor

        # ── Adaptive temporal thresholds (always derived from FPS) ────────
        adaptive = compute_adaptive_thresholds(fps)
        self.registry   = DamageRegistry(
            min_votes=adaptive["min_votes"],
            min_ratio=adaptive["min_ratio"],
        )

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

        # ── Step b: Use raw frame direction ───────────────────────────────────
        stable_dir = car_direction

        # ── Step c: Context-aware part list ───────────────────────────────────
        allowed_parts = CAR_PARTS_MAP.get(stable_dir, [])
        if not allowed_parts:
            # Unknown direction → skip this frame safely
            return parts_frame, damage_frame, stable_dir

        # ── Step d: Part segmentation (filtered) ──────────────────────────────
        detected_parts = self.parts_seg.predict(
            frame, allowed_parts, self.parts_conf_floor
        )

        # Deduplicate only near-identical same-class boxes. Keeping one item per
        # class would drop valid paired parts such as headlights, tail-lights,
        # and wheels.
        def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                return 0.0
            area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
            area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            denom = area_a + area_b - inter
            return inter / denom if denom > 0 else 0.0

        filtered_parts: List[Dict] = []
        for part in sorted(detected_parts, key=lambda p: p["conf"], reverse=True):
            duplicate = any(
                part["part_name"] == kept["part_name"]
                and _bbox_iou(part["bbox"], kept["bbox"]) > 0.85
                for kept in filtered_parts
            )
            if not duplicate:
                filtered_parts.append(part)
        detected_parts = filtered_parts

        drawings: List[Dict] = []

        for part_info in detected_parts:
            part_name  = part_info["part_name"]
            part_conf  = part_info["conf"]
            x1, y1, x2, y2 = part_info["bbox"]
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, min(frame_w, x1))
            x2 = max(0, min(frame_w, x2))
            y1 = max(0, min(frame_h, y1))
            y2 = max(0, min(frame_h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            part_info["bbox"] = (x1, y1, x2, y2)
            mask_pts   = part_info["mask_pts"]
            color      = part_color(part_name)

            # Compute part mask area in pixels (for severity ratio denominator).
            # Falls back to bounding-box area when the mask polygon is unavailable.
            if mask_pts is not None and mask_pts.size >= 6:
                part_info["part_area_px"] = float(cv2.contourArea(mask_pts.astype(np.float32)))
            else:
                part_info["part_area_px"] = float((x2 - x1) * (y2 - y1))

            # Mark that this part was visible (for ratio denominator).
            # stable_dir is passed so the counter is tracked per (part, direction)
            # key — keeping the vote denominator accurate for each directional view.
            self.registry.mark_part_seen(track_id, part_name, stable_dir, frame_index,
                                          part_area_px=part_info["part_area_px"])

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
            crop_w = x2 - x1
            crop_h = y2 - y1
            valid_dmg_idx = 0
            for d_type, d_conf, d_mask_pts, d_bbox in dmg_results:
                # Calculate damage box in frame coordinates
                dx1 = x1 + d_bbox[0]
                dy1 = y1 + d_bbox[1]
                dx2 = x1 + d_bbox[2]
                dy2 = y1 + d_bbox[3]
                dcx = int((dx1 + dx2) / 2)
                dcy = int((dy1 + dy2) / 2)

                # Re-attribute: find which part's mask actually contains/is closest to the damage centroid
                assigned_part = None
                best_dist = -9999.0
                for p_info in detected_parts:
                    p_mask = p_info["mask_pts"]
                    if p_mask is not None and p_mask.size > 0:
                        dist = cv2.pointPolygonTest(p_mask, (dcx, dcy), True)
                        if dist > best_dist:
                            best_dist = dist
                            assigned_part = p_info

                # If masks were available and the best matching part is too far
                # outside every part, skip it. When masks are absent, fall back
                # to the current crop instead of dropping all detections.
                if assigned_part is not None and best_dist < -15.0:
                    continue

                # If we successfully matched to a part, use its info
                if assigned_part is not None:
                    target_part_name = assigned_part["part_name"]
                    target_color = part_color(target_part_name)
                    target_location = resolve_damage_location(target_part_name, stable_dir)
                    ax1, ay1, ax2, ay2 = assigned_part["bbox"]
                    acrop_w = ax2 - ax1
                    acrop_h = ay2 - ay1
                    # Translate damage bbox to be relative to the assigned part's crop
                    target_bbox = (dx1 - ax1, dy1 - ay1, dx2 - ax1, dy2 - ay1)
                    target_part_area_px = assigned_part.get("part_area_px", 0.0)
                else:
                    # Fallback to the current crop's part
                    target_part_name = part_name
                    target_color = color
                    target_location = location
                    acrop_w = crop_w
                    acrop_h = crop_h
                    target_bbox = d_bbox
                    target_part_area_px = part_info.get("part_area_px", 0.0)

                # Re-attribution guard: after changing the owning part, verify
                # that d_type is still physically valid for that part.
                # Example: a scratch detected on a Fender crop (fender bbox bleeds
                # over the tire) whose centroid lands inside the wheel polygon would
                # otherwise be recorded as "Front-wheel: scratch" — but the wheel
                # only supports "flat_tire".  Drop such cross-part bleed-over hits.
                if d_type not in get_allowed_damage(target_part_name):
                    log.debug(
                        "Re-attribution guard: skipping '%s' on '%s' "
                        "(not in allowed damage types for that part).",
                        d_type, target_part_name,
                    )
                    continue

                # Keep a copy of the crop-relative mask for the registry
                # (centroid computation via cv2.moments requires crop-relative coords).
                # d_mask_pts is currently in the coordinate space of the original
                # part crop (x1, y1); we re-relativise it to the assigned part's
                # crop origin (ax1, ay1) when re-attribution has occurred.
                crop_rel_mask: Optional[np.ndarray] = None
                if d_mask_pts is not None and d_mask_pts.size >= 6:
                    if assigned_part is not None:
                        # d_mask_pts is crop-relative to (x1, y1).
                        # The assigned part crop starts at (ax1, ay1).
                        # Shift so it is relative to the assigned crop's origin.
                        ax1_r, ay1_r, _, _ = assigned_part["bbox"]
                        offset = np.array([ax1_r - x1, ay1_r - y1], dtype=np.int32)
                        crop_rel_mask = (d_mask_pts - offset).astype(np.int32)
                    else:
                        # No re-attribution — mask is already relative to (x1, y1)
                        crop_rel_mask = d_mask_pts.copy()

                # Translate crop-relative mask coordinates to frame coordinates
                # (used only for on-screen drawing below).
                if d_mask_pts is not None:
                    d_mask_pts = (d_mask_pts + np.array([x1, y1])).astype(np.int32)

                # Update the temporal registry with spatial instance tracking.
                # Pass the crop-relative mask so the registry can compute the
                # true mask centroid instead of the bounding-box centroid.
                self.registry.update(
                    track_id=track_id,
                    part_name=target_part_name,
                    damage_type=d_type,
                    confidence=d_conf,
                    car_direction=stable_dir,
                    frame_index=frame_index,
                    damage_mask=crop_rel_mask,
                    damage_bbox=target_bbox,
                    crop_size=(acrop_w, acrop_h),
                    location=target_location,
                    part_area_px=target_part_area_px,
                )

                drawings.append({
                    "x1": dx1, "y1": dy1, "x2": dx2, "y2": dy2,
                    "part_name":  target_part_name,
                    "color":      target_color,
                    "d_mask":     d_mask_pts,
                    "d_type":     d_type,
                    "d_conf":     d_conf,
                    "location":   target_location,
                    "label_idx":  valid_dmg_idx,
                })
                valid_dmg_idx += 1

        # Blend part segmentation masks
        if detected_parts:
            cv2.addWeighted(overlay_p, 0.30, parts_frame, 0.70, 0, parts_frame)

        # Annotate damage frame.
        # Multiple damage types on the same part share the same bbox anchor,
        # so we stack their labels upward using label_idx to avoid overlap.
        _LABEL_LINE_H = 18  # vertical step between stacked damage labels (px)
        for d in drawings:
            if d["d_mask"] is not None and d["d_mask"].size > 0:
                # Draw the damage segmentation polygon contour & semi-transparent fill
                overlay_d = damage_frame.copy()
                cv2.fillPoly(overlay_d, [d["d_mask"]], (0, 0, 255))
                cv2.polylines(damage_frame, [d["d_mask"]],
                              isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.addWeighted(overlay_d, 0.40, damage_frame, 0.60, 0, damage_frame)
            else:
                # Fallback: draw damage bounding box if mask is not available
                cv2.rectangle(damage_frame,
                              (d["x1"], d["y1"]), (d["x2"], d["y2"]),
                              d["color"], 2)
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
    video_path:   str,
    output_path:  str  = "result_pipeline.mp4",
    parts_conf:   float = 0.30,
    damage_conf:  float = 0.30,
    frame_skip:   int   = 1,
    preview:      bool  = False,
    save_parts:   bool  = True,
    save_damage:  bool  = True,
    report_path:  Optional[str] = None,
    debug:        bool  = False,
) -> None:
    """
    Full end-to-end pipeline runner for a video file.

    Writes two annotated output videos (_parts, _damage) and optionally
    saves a JSON damage report with the finalized registry output.

    Temporal voting thresholds (min_votes, direction_streak) are derived
    automatically from the video's FPS via compute_adaptive_thresholds().
    """
    validate_runtime_options(parts_conf, damage_conf)
    if frame_skip < 1:
        raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")

    log.info("=" * 65)
    log.info("Car Damage Detection — Integrated 3-Model Pipeline")
    log.info("Input  : %s", video_path)
    log.info("Output : %s", output_path)
    log.info("=" * 65)

    # ── Read video metadata FIRST so adaptive thresholds can use real FPS ─
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Effective FPS after frame-skipping: vote windows stay calibrated
    # at the rate frames are actually processed, not the raw capture rate.
    effective_fps = src_fps / frame_skip
    out_fps       = effective_fps

    log.info("Resolution : %dx%d   FPS: %.1f → %.1f (effective)",
             width, height, src_fps, effective_fps)

    # ── Build pipeline — all temporal thresholds auto-derived from FPS ────
    pipeline = CarDamagePipeline(
        parts_conf_floor=parts_conf,
        damage_conf_floor=damage_conf,
        fps=effective_fps,
    )
    log.info("Frames     : %d  (every %d)", total_frames, frame_skip)

    base   = os.path.splitext(output_path)[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer_parts: Optional[cv2.VideoWriter] = None
    if save_parts:
        ensure_parent_dir(f"{base}_parts.mp4")
        writer_parts = cv2.VideoWriter(
            f"{base}_parts.mp4", fourcc, out_fps, (width, height)
        )
        if not writer_parts.isOpened():
            log.warning("VideoWriter for parts could not be opened — parts video will be skipped.")
            writer_parts = None

    writer_damage: Optional[cv2.VideoWriter] = None
    if save_damage:
        ensure_parent_dir(f"{base}_damage.mp4")
        writer_damage = cv2.VideoWriter(
            f"{base}_damage.mp4", fourcc, out_fps, (width, height)
        )
        if not writer_damage.isOpened():
            log.warning("VideoWriter for damage could not be opened — damage video will be skipped.")
            writer_damage = None

    frame_idx = written = error_count = 0
    max_frame_errors = 10
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
                error_count += 1
                log.exception("Frame %d failed during pipeline processing.", frame_idx)
                if error_count >= max_frame_errors:
                    raise RuntimeError(
                        f"Aborting after {error_count} frame processing errors. "
                        "Check model compatibility and input video quality."
                    ) from exc
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
    if written == 0:
        if error_count > 0:
            raise RuntimeError(
                f"No frames were processed successfully; {error_count} frame errors occurred."
            )
        raise RuntimeError("No frames were processed. Check frame_skip and the input video.")

    if debug:
        print(pipeline.registry.debug_registry())

    # Call finalize() exactly once so that PartRecord.confirmed_damages is
    # written only once.  Both the console summary and the optional JSON file
    # share the same pre-computed result — no redundant re-voting.
    report = pipeline.registry.finalize()
    print(DamageRegistry.format_report(report))

    if report_path:
        ensure_parent_dir(report_path)
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
    • DamageRegistry is created with min_votes=1 so
      that a single-frame detection counts as "confirmed" — temporal
      voting only makes sense across multiple frames.
    • Output is two annotated images (_parts / _damage) instead of videos.
    """
    validate_runtime_options(parts_conf, damage_conf)

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
    # immediately confirmed.  Passing fps=1.0 makes compute_adaptive_thresholds
    # produce min_votes=1 and direction_streak=1 via the normal formula.
    pipeline = CarDamagePipeline(
        parts_conf_floor=parts_conf,
        damage_conf_floor=damage_conf,
        fps=1.0,
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

    ensure_parent_dir(parts_out)
    ensure_parent_dir(damage_out)
    if not cv2.imwrite(parts_out, parts_frm):
        raise RuntimeError(f"Failed to write parts image: {parts_out}")
    if not cv2.imwrite(damage_out, damage_frm):
        raise RuntimeError(f"Failed to write damage image: {damage_out}")
    log.info("Parts image  → %s", parts_out)
    log.info("Damage image → %s", damage_out)

    if report_path:
        ensure_parent_dir(report_path)
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
  python pipeline.py car.jpg --parts-conf 0.25 --damage-conf 0.25  
        """,
    )
    ap.add_argument("input",
                    help="Input file — video (mp4/avi/…) or image (jpg/png/bmp/…).\n"
                         "Mode is auto-detected from the file extension.")
    ap.add_argument("--output",      default="result_pipeline",
                    help="Base output path without extension (default: result_pipeline).\n"
                         "Extensions are added automatically (_parts.mp4 / _parts.jpg etc.).")
    ap.add_argument("--parts-conf",  type=float, default=0.50,
                    help="Parts segmentation conf floor (default: 0.50)")
    ap.add_argument("--damage-conf", type=float, default=0.50,
                    help="Damage detection conf floor (default: 0.50)")
    # ── Video-only flags ───────────────────────────────────────────────────────
    ap.add_argument("--frame-skip",  type=int, default=1,
                    help="[Video only] Process every Nth frame (default: 1)")
    ap.add_argument("--preview",     action="store_true",
                    help="[Video only] Show live preview windows (press q to stop)")
    ap.add_argument("--no-parts",    action="store_true",
                    help="[Video only] Skip parts output video")
    ap.add_argument("--no-damage",   action="store_true",
                    help="[Video only] Skip damage output video")
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
            debug       = args.debug,
        )
