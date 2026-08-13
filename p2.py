"""
damage_pipeline.py — Two-Pass Keyframe Car Damage Detection
===================================================================
Instead of processing every frame with all three models and voting,
this pipeline uses a fast two-pass strategy:

    Pass 1 (fast):   Scan video with the angle classifier only.
                     Pick the top N best frames per direction (up to 8 directions).

    Pass 2 (targeted): Run parts segmentation + damage detection on
                       only those keyframes.

Result: ~10-50× faster than frame-by-frame, with cleaner results
since each direction gets deliberately-selected, high-quality frames.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("damage_pipeline")

BASE_DIR = Path(__file__).resolve().parent

# ── Model weights — edit these paths for your environment ───────────────────
MODEL_ANGLE_PATH  = BASE_DIR / "models" / "car_angle.pt"
MODEL_PARTS_PATH  = BASE_DIR / "models" / "car_part.pt"
MODEL_DAMAGE_PATH = BASE_DIR / "models" / "damage_type_seg_6classes.pt"


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    angle_model_path: str = str(MODEL_ANGLE_PATH)
    parts_model_path: str = str(MODEL_PARTS_PATH)
    damage_model_path: str = str(MODEL_DAMAGE_PATH)

    parts_conf: float = 0.60
    damage_conf: float = 0.60

    # How crops are built before being handed to the damage model.
    #   "bbox"   -> raw axis-aligned crop (original behaviour)
    crop_strategy: str = "bbox"

    # How often to sample frames during the angle-scan pass.
    # E.g. 5 means check every 5th frame for direction classification.
    sample_every_n: int = 1

    # Minimum number of frames that must classify to the same direction
    # before that direction is considered "confirmed".  This filters out
    # one-off misclassifications — if only 1 random frame says "front"
    # but the angle model never sees "front" again, it's likely wrong.
    min_direction_frames: int = 1

    # How many frames to analyze per confirmed direction.
    # More frames = better coverage (catches damage missed by one angle),
    # but slightly slower.  E.g. 7 means ~56 total frames across 8 directions.
    frames_per_direction: int = 7

    # Save the selected keyframe images to disk for inspection.
    save_keyframes: bool = True

    draw: bool = True

    # Minimum Laplacian variance (sharpness) for a frame to be considered
    # as a keyframe candidate.  Frames below this are blurry / out-of-focus
    # and will produce poor segmentation masks.  Set to 0 to disable.
    min_sharpness: float = 50.0


# Camera-view -> car-centric direction. The classifier looks *at* the car;
# this remaps to how the car itself is oriented.
CAMERA_TO_CAR_DIRECTION: Dict[str, str] = {
    "front-right": "front-left-side",
    "front-left": "front-right-side",
    "back-right": "back-left-side",
    "back-left": "back-right-side",
    "side-right": "left-side",
    "side-left": "right-side",
    "front": "front",
    "back": "back",
}

PARTS_VISIBLE_FROM: Dict[str, List[str]] = {
    "front": ["Front-bumper", "Headlight", "Hood", "Windshield"],
    "front-left-side": ["Front-bumper", "Fender", "Mirror", "Headlight", "Windshield"],
    "front-right-side": ["Front-bumper", "Fender", "Mirror", "Headlight", "Windshield"],
    "back": ["Back-bumper", "Trunk", "Tail-light", "Back-windshield"],
    "back-left-side": ["Back-bumper", "Quarter-panel", "Tail-light", "Back-windshield"],
    "back-right-side": ["Back-bumper", "Quarter-panel", "Tail-light", "Back-windshield"],
    "left-side": ["Front-door", "Back-door", "Front-wheel", "Back-wheel", "Front-window", "Back-window","Fender", "Quarter-panel", "Mirror", "Rocker-panel"],
    "right-side": ["Front-door", "Back-door", "Front-wheel", "Back-wheel", "Front-window","Back-window", "Fender", "Quarter-panel", "Mirror", "Rocker-panel"],
}

DAMAGE_ALLOWED_ON_PART: Dict[str, List[str]] = {
    "Front-wheel": ["flat_tire"], 
    "Back-wheel": ["flat_tire"],
    "Windshield": ["glass_break"], 
    "Back-windshield": ["glass_break"],
    "Headlight": ["broken_light"], 
    "Tail-light": ["broken_light"],
    "Mirror": ["crack", "scratch"],
    "Front-bumper": ["dent", "scratch", "crack"], 
    "Back-bumper": ["dent", "scratch", "crack"],
    "Hood": ["dent", "scratch", "crack"], 
    "Trunk": ["dent", "scratch", "crack"],
    "Fender": ["dent", "scratch", "crack"], 
    "Front-door": ["dent", "scratch", "crack"],
    "Back-door": ["dent", "scratch", "crack"], 
    "Quarter-panel": ["dent", "scratch", "crack"],
    "Rocker-panel": ["dent", "scratch", "crack"],
    "Front-window": ["glass_break"],
    "Back-window": ["glass_break"],
}

SEVERITY_BANDS: List[Tuple[float, str]] = [(0.05, "Minor"), (0.20, "Moderate"), (1.01, "Severe")]

# Singular car parts that span across multiple adjacent view angles
SINGULAR_PARTS: set = {"Front-bumper", "Back-bumper", "Hood", "Trunk", "Windshield", "Back-windshield"}

# Adjacent views for cross-view deduplication
VIEW_ADJACENCY: Dict[str, set] = {
    "front": {"front-left-side", "front-right-side"},
    "back": {"back-left-side", "back-right-side"},
    "left-side": {"front-left-side", "back-left-side"},
    "right-side": {"front-right-side", "back-right-side"},
    "front-left-side": {"front", "left-side"},
    "front-right-side": {"front", "right-side"},
    "back-left-side": {"back", "left-side"},
    "back-right-side": {"back", "right-side"},
}


def severity_for(ratio: float) -> str:
    for cutoff, label in SEVERITY_BANDS:
        if ratio <= cutoff:
            return label
    return "Severe"


def calculate_sharpness(frame: np.ndarray) -> float:
    """Return Laplacian variance as a sharpness score.

    Higher = sharper.  Blurry / motion-blurred frames typically score < 50,
    while in-focus frames score 100+.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ═══════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT DATA CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PartBox:
    name: str
    conf: float
    xyxy: Tuple[int, int, int, int]
    mask_xy: Optional[np.ndarray]  # frame-relative polygon, or None
    area_px: float


@dataclass
class DamageBox:
    dtype: str
    conf: float
    xyxy_in_crop: Tuple[int, int, int, int]
    mask_in_crop: Optional[np.ndarray]


# ═══════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS — one thin class per stage, nothing more than "run + parse"
# ═══════════════════════════════════════════════════════════════════════════

class AngleStage:
    def __init__(self, weights: str):
        self.model = YOLO(weights)

    def infer(self, frame: np.ndarray) -> Tuple[str, float]:
        r = self.model.predict(frame, verbose=False)[0]
        cls_id = r.probs.top1
        return r.names[cls_id], float(r.probs.top1conf)


class PartsStage:
    def __init__(self, weights: str):
        self.model = YOLO(weights)

    def infer(self, frame: np.ndarray, allowed: List[str], conf: float) -> List[PartBox]:
        r = self.model.predict(frame, conf=conf, verbose=False)[0]
        out: List[PartBox] = []
        h, w = frame.shape[:2]
        for i, box in enumerate(r.boxes):
            name = r.names[int(box.cls[0])]
            if name not in allowed:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            mask_xy = None
            area_px = float((x2 - x1) * (y2 - y1))
            if r.masks is not None and i < len(r.masks.xy):
                mask_xy = r.masks.xy[i].astype(np.int32)
                if mask_xy.size >= 6:
                    area_px = float(cv2.contourArea(mask_xy.astype(np.float32)))
            out.append(PartBox(name, float(box.conf[0]), (x1, y1, x2, y2), mask_xy, area_px))
        return out


class DamageStage:
    def __init__(self, weights: str):
        self.model = YOLO(weights)

    def infer(self, crop: np.ndarray, allowed: List[str], conf: float) -> List[DamageBox]:
        if crop.size == 0 or not allowed:
            return []
        r = self.model.predict(crop, conf=conf, verbose=False)[0]
        raw: List[DamageBox] = []
        for i, box in enumerate(r.boxes):
            dtype = r.names[int(box.cls[0])]
            if dtype not in allowed:
                continue
            xyxy = tuple(map(int, box.xyxy[0]))
            mask = None
            if r.masks is not None and i < len(r.masks.xy):
                mask = r.masks.xy[i].astype(np.int32)
            raw.append(DamageBox(dtype, float(box.conf[0]), xyxy, mask))
        return _dedupe_same_type(raw, crop.shape[:2])


def _dedupe_same_type(candidates: List[DamageBox], crop_hw: Tuple[int, int]) -> List[DamageBox]:
    """Collapse duplicate detections of the same damage type via mask/box IoU."""
    def iou(a: DamageBox, b: DamageBox) -> float:
        if a.mask_in_crop is not None and b.mask_in_crop is not None and a.mask_in_crop.size >= 6 and b.mask_in_crop.size >= 6:
            h, w = crop_hw
            ma = np.zeros((h, w), np.uint8); mb = np.zeros((h, w), np.uint8)
            cv2.fillPoly(ma, [a.mask_in_crop], 1); cv2.fillPoly(mb, [b.mask_in_crop], 1)
            union = np.logical_or(ma, mb).sum()
            return float(np.logical_and(ma, mb).sum()) / union if union else 0.0
        ax1, ay1, ax2, ay2 = a.xyxy_in_crop; bx1, by1, bx2, by2 = b.xyxy_in_crop
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union else 0.0

    kept: List[DamageBox] = []
    for cand in sorted(candidates, key=lambda c: c.conf, reverse=True):
        if any(cand.dtype == k.dtype and iou(cand, k) > 0.5 for k in kept):
            continue
        kept.append(cand)
    return kept


# ═══════════════════════════════════════════════════════════════════════════
# CROPPING — the swappable part discussed earlier in this conversation
# ═══════════════════════════════════════════════════════════════════════════

def build_crop(frame: np.ndarray, part: PartBox, cfg: PipelineConfig) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Return (crop_image, (origin_x, origin_y)) for damage inference."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = part.xyxy

    if cfg.crop_strategy == "matte" and part.mask_xy is not None and part.mask_xy.size >= 6:
        crop = frame[y1:y2, x1:x2].copy()
        local_mask = (part.mask_xy - np.array([x1, y1])).astype(np.int32)
        alpha = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(alpha, [local_mask], 255)
        crop[alpha == 0] = 0
        return crop, (x1, y1)

    # default: plain bbox
    return frame[y1:y2, x1:x2], (x1, y1)


# ═══════════════════════════════════════════════════════════════════════════
# PART RE-ATTRIBUTION — which part does a damage box really belong to?
# ═══════════════════════════════════════════════════════════════════════════

def reattribute(dmg: DamageBox, origin: Tuple[int, int], parts: List[PartBox]) -> Optional[PartBox]:
    """Re-attribute damage to whichever detected part's mask or bounding box
    actually contains/overlaps the damage best in frame coordinates, provided
    the damage type is allowed on that target part.
    If the damage physically lies far outside any allowed/detected part, returns None."""
    ox, oy = origin
    dx1, dy1, dx2, dy2 = dmg.xyxy_in_crop
    fx1, fy1, fx2, fy2 = ox + dx1, oy + dy1, ox + dx2, oy + dy2
    damage_center = (int((fx1 + fx2) / 2), int((fy1 + fy2) / 2))

    best_part = None
    best_score = -1e9

    for p in parts:
        # Must be physically allowed on this part
        if dmg.dtype not in DAMAGE_ALLOWED_ON_PART.get(p.name, []):
            continue

        score = -1e9
        # 1. Mask polygon containment test (if part has polygon mask)
        if p.mask_xy is not None and p.mask_xy.size >= 6:
            poly_dist = cv2.pointPolygonTest(p.mask_xy, damage_center, True)
            if poly_dist >= 0:
                score = 100.0 + poly_dist
            else:
                score = poly_dist
        else:
            # 2. Bounding box containment test (if part only has bbox)
            px1, py1, px2, py2 = p.xyxy
            if px1 <= damage_center[0] <= px2 and py1 <= damage_center[1] <= py2:
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                dist = ((damage_center[0] - pcx)**2 + (damage_center[1] - pcy)**2)**0.5
                score = 50.0 - dist
            else:
                ix1, iy1 = max(fx1, px1), max(fy1, py1)
                ix2, iy2 = min(fx2, px2), min(fy2, py2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    score = inter_area / float((fx2 - fx1) * (fy2 - fy1))

        if score > best_score:
            best_score = score
            best_part = p

    # If the damage center is within ~5 pixels of the best valid part, accept it.
    if best_part is not None and best_score >= -5.0:
        return best_part

    # Otherwise, it does not physically belong to ANY detected/allowed part in this view.
    return None


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — Two-Pass Keyframe Pipeline
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Keyframe:
    """A selected frame for one direction."""
    direction: str
    confidence: float
    frame_idx: int
    timestamp_seconds: float = 0.0
    frame: Optional[np.ndarray] = None  # Loaded lazily to save memory


class DamagePipeline:
    def __init__(self, cfg: PipelineConfig):
        log.info("Loading angle / parts / damage models …")
        self.cfg = cfg
        self.angle = AngleStage(cfg.angle_model_path)
        self.parts = PartsStage(cfg.parts_model_path)
        self.damage = DamageStage(cfg.damage_model_path)
        log.info("Models ready.")

    # ── Pass 1: Scan video, pick best frames per direction ───────────────

    def scan_video(self, video_path: str) -> Dict[str, List[Keyframe]]:
        """Run only the angle classifier on sampled frames.

        Collects ALL candidates per direction, confirms a direction only if
        it was classified by at least `min_direction_frames` frames, then
        picks the top `frames_per_direction` frames spread apart in time.

        Returns {direction: [Keyframe, ...]} with up to N keyframes each.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        # Collect every candidate: direction -> list of (conf, frame_idx)
        candidates: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

        frame_idx = 0
        scanned = 0

        pbar = tqdm(
            total=total_frames if total_frames > 0 else None,
            desc="[Pass 1] Scanning frames",
            unit="frame",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % self.cfg.sample_every_n != 0:
                pbar.update(1)
                continue

            raw_view, conf = self.angle.infer(frame)
            car_direction = CAMERA_TO_CAR_DIRECTION.get(raw_view, raw_view)
            scanned += 1

            # Quality gate: skip blurry / out-of-focus frames
            if self.cfg.min_sharpness > 0:
                sharpness = calculate_sharpness(frame)
                if sharpness < self.cfg.min_sharpness:
                    pbar.update(1)
                    continue

            candidates[car_direction].append((conf, frame_idx))
            pbar.update(1)
            pbar.set_postfix_str(f"{len(candidates)} dirs seen", refresh=False)

        pbar.close()
        cap.release()

        # Filter + select top N frames per confirmed direction
        min_frames = self.cfg.min_direction_frames
        n_pick = self.cfg.frames_per_direction
        confirmed: Dict[str, List[Keyframe]] = {}

        for direction, entries in sorted(candidates.items()):
            count = len(entries)
            if count < min_frames:
                log.warning("  %-20s REJECTED — only %d frame(s), need %d",
                            direction, count, min_frames)
                continue

            # Select top N frames, spread apart in time
            selected = self._pick_spread_frames(entries, n_pick)
            confirmed[direction] = [
                Keyframe(direction=direction,
                         confidence=conf, frame_idx=fidx,
                         timestamp_seconds=round(fidx / fps, 2))
                for conf, fidx in selected
            ]

        log.info("Pass 1 done: scanned %d frames, confirmed %d/%d directions "
                 "(min_direction_frames=%d, frames_per_direction=%d).",
                 scanned, len(confirmed), len(candidates), min_frames, n_pick)
        for d, kfs in sorted(confirmed.items()):
            total = len(candidates[d])
            frames_str = ", ".join(f"#{kf.frame_idx}" for kf in kfs)
            log.info("  %-20s %d frame(s) selected [%s]  (%d total agreed)",
                     d, len(kfs), frames_str, total)
        return confirmed

    @staticmethod
    def _pick_spread_frames(
        entries: List[Tuple[float, int]], n: int
    ) -> List[Tuple[float, int]]:
        """Pick up to `n` frames from `entries`, preferring high confidence
        while spreading them apart in time so we don't get N near-identical
        frames from the same moment."""
        if len(entries) <= n:
            return entries

        # Sort by confidence descending
        by_conf = sorted(entries, key=lambda e: e[0], reverse=True)

        # Greedily pick frames that are temporally spread apart
        selected: List[Tuple[float, int]] = []
        # Minimum frame gap: total span / (n+1) so they're spaced out
        frame_indices = [e[1] for e in entries]
        span = max(frame_indices) - min(frame_indices)
        min_gap = max(1, span // (n + 1))

        for entry in by_conf:
            if len(selected) >= n:
                break
            fidx = entry[1]
            # Check temporal distance to already-selected frames
            if all(abs(fidx - s[1]) >= min_gap for s in selected):
                selected.append(entry)

        # If we couldn't fill N due to gap constraint, relax and fill remaining
        if len(selected) < n:
            for entry in by_conf:
                if len(selected) >= n:
                    break
                if entry not in selected:
                    selected.append(entry)

        return selected

    # ── Load keyframe pixels from video (lazy retrieval) ─────────────────

    def _load_keyframe_pixels(self, video_path: str, keyframes: Dict[str, List[Keyframe]]) -> None:
        """Re-open the video and read only the selected keyframe pixels.

        This avoids storing all scanned frames in memory during Pass 1.
        Frames are read in ascending index order for sequential I/O.
        """
        idx_to_keyframes: Dict[int, List[Keyframe]] = defaultdict(list)
        for kf_list in keyframes.values():
            for kf in kf_list:
                idx_to_keyframes[kf.frame_idx].append(kf)

        if not idx_to_keyframes:
            return

        sorted_indices = sorted(idx_to_keyframes.keys())
        log.info("Loading %d keyframe pixels from video …", len(sorted_indices))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot re-open video for keyframe loading: {video_path}")

        for target_idx in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx - 1)  # 0-based seek
            ok, frame = cap.read()
            if not ok:
                log.warning("Failed to read frame #%d from video", target_idx)
                continue
            # Multiple keyframes may share the same frame index
            for kf in idx_to_keyframes[target_idx]:
                kf.frame = frame.copy()

        cap.release()
        log.info("Keyframe pixels loaded.")

    # ── Pass 2: Run parts + damage on selected keyframes ─────────────────

    def _draw_part(self, vis: np.ndarray, part: PartBox) -> None:
        """Render car part as a polyline segmentation outline if available, falling back to box."""
        if part.mask_xy is not None and part.mask_xy.size >= 6:
            cv2.polylines(vis, [part.mask_xy], isClosed=True, color=(255, 180, 0), thickness=2)
            label_x = int(part.mask_xy[:, 0].min())
            label_y = int(part.mask_xy[:, 1].min())
        else:
            px1, py1, px2, py2 = part.xyxy
            cv2.rectangle(vis, (px1, py1), (px2, py2), (255, 180, 0), 1)
            label_x, label_y = px1, py1

        cv2.putText(vis, part.name, (label_x, max(15, label_y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

    def _draw_damage(self, vis: np.ndarray, part_name: str, dtype: str, conf: float,
                      poly_str: Optional[str], bbox: Tuple[int, int, int, int]) -> None:
        """Render damage as a filled, semi-transparent segmentation mask."""
        if poly_str:
            poly = np.array(list(map(float, poly_str.split()))).reshape(-1, 2).astype(np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [poly], (0, 0, 255))
            cv2.polylines(vis, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.addWeighted(overlay, 0.40, vis, 0.60, 0, vis)
            label_x, label_y = poly[:, 0].min(), poly[:, 1].min()
        else:
            dx1, dy1, dx2, dy2 = bbox
            cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
            label_x, label_y = dx1, dy1

        cv2.putText(vis, f"{part_name}:{dtype} {conf:.2f}",
                    (int(label_x), max(15, int(label_y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    def analyze_keyframes(self, keyframes: Dict[str, List[Keyframe]]) -> List[Dict]:
        """Run parts + damage detection on all keyframes per direction.

        Processes every selected frame, then deduplicates: if the same
        (part, damage_type) is found on multiple frames of the same
        direction and they are physically close, only the highest-confidence hit is kept.

        Returns a list of raw damage hits (with 'vis_image' attached if drawing is enabled).
        """
        report: List[Dict] = []
        total_frames = sum(len(kfs) for kfs in keyframes.values())

        pbar = tqdm(
            total=total_frames,
            desc="[Pass 2] Analyzing keyframes",
            unit="frame",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        )

        for direction, kf_list in sorted(keyframes.items()):
            # Per-direction raw hits (before dedup)
            dir_hits: List[Dict] = []

            for kf in kf_list:
                pbar.update(1)
                pbar.set_postfix_str(f"{direction} #{kf.frame_idx}", refresh=False)

                frame = kf.frame
                allowed_parts = PARTS_VISIBLE_FROM.get(direction, [])
                detected_parts = self.parts.infer(frame, allowed_parts, self.cfg.parts_conf)

                for part in detected_parts:
                    crop, (ox, oy) = build_crop(frame, part, self.cfg)
                    if crop.size == 0:
                        continue

                    # Union of allowed damage types for all parts overlapping this crop box
                    crop_x1, crop_y1, crop_x2, crop_y2 = part.xyxy
                    overlapping_parts = [
                        p for p in detected_parts
                        if max(crop_x1, p.xyxy[0]) < min(crop_x2, p.xyxy[2]) and max(crop_y1, p.xyxy[1]) < min(crop_y2, p.xyxy[3])
                    ]
                    allowed_dmg = sorted(list({
                        dtype
                        for p in overlapping_parts
                        for dtype in DAMAGE_ALLOWED_ON_PART.get(p.name, [])
                    }))
                    if not allowed_dmg:
                        allowed_dmg = DAMAGE_ALLOWED_ON_PART.get(part.name, [])

                    for dmg in self.damage.infer(crop, allowed_dmg, self.cfg.damage_conf):
                        dx1, dy1, dx2, dy2 = dmg.xyxy_in_crop
                        target = reattribute(dmg, origin=(ox, oy), parts=detected_parts)
                        if target is None:
                            continue

                        # Translate damage bbox back to full-frame coordinates
                        fx1, fy1, fx2, fy2 = ox + dx1, oy + dy1, ox + dx2, oy + dy2

                        damage_poly_str = None
                        if dmg.mask_in_crop is not None and dmg.mask_in_crop.size >= 6:
                            poly = (dmg.mask_in_crop + np.array([ox, oy])).astype(np.int32)

                            # COOKIE-CUTTER: Trim the damage polygon to the target part's polygon
                            if target.mask_xy is not None and target.mask_xy.size >= 6:
                                h, w = frame.shape[:2]
                                dmg_canvas = np.zeros((h, w), dtype=np.uint8)
                                part_canvas = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillPoly(dmg_canvas, [poly], 255)
                                cv2.fillPoly(part_canvas, [target.mask_xy], 255)

                                intersection = cv2.bitwise_and(dmg_canvas, part_canvas)
                                contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                if contours:
                                    best_contour = max(contours, key=cv2.contourArea)
                                    if len(best_contour) >= 3:
                                        poly = best_contour.reshape(-1, 2)
                                        # Update the damage bbox to match the new trimmed mask
                                        x, y, w_box, h_box = cv2.boundingRect(poly)
                                        fx1, fy1, fx2, fy2 = x, y, x + w_box, y + h_box
                                    else:
                                        continue  # Intersection was just a tiny sliver, ignore this damage
                                else:
                                    continue  # Damage is completely outside the part's polygon

                            damage_area = float(cv2.contourArea(poly.astype(np.float32)))
                            damage_poly_str = " ".join(map(str, poly.reshape(-1).tolist()))
                        else:
                            damage_area = float((fx2 - fx1) * (fy2 - fy1))

                        # Compute severity from (possibly trimmed) damage area vs part area
                        sev_ratio = damage_area / target.area_px if target.area_px > 0 else 0.0

                        part_poly_str = None
                        if target.mask_xy is not None and target.mask_xy.size >= 6:
                            part_poly_str = " ".join(map(str, target.mask_xy.reshape(-1).tolist()))

                        # Compute relative centroid of damage within the part bbox
                        px1, py1, px2, py2 = target.xyxy
                        part_w = max(px2 - px1, 1)
                        part_h = max(py2 - py1, 1)
                        dmg_cx = (fx1 + fx2) / 2.0
                        dmg_cy = (fy1 + fy2) / 2.0
                        rel_cx = (dmg_cx - px1) / part_w
                        rel_cy = (dmg_cy - py1) / part_h

                        hit = {
                            "part": target.name,
                            "damage_type": dmg.dtype,
                            "car_view": direction,
                            "confidence": round(dmg.conf, 3),
                            "severity": severity_for(sev_ratio),
                            "severity_ratio": round(sev_ratio, 4),
                            "frame_index": kf.frame_idx,
                            "timestamp_seconds": kf.timestamp_seconds,
                            "part_bbox": " ".join(map(str, target.xyxy)),
                            "part_polygon": part_poly_str,
                            "damage_bbox": " ".join(map(str, [fx1, fy1, fx2, fy2])),
                            "damage_polygon": damage_poly_str,
                            "_rel_centroid": (rel_cx, rel_cy)
                        }

                        if self.cfg.draw:
                            # Create a fresh copy of the frame to draw all parts + damage
                            part_vis = frame.copy()
                            # 1. Draw ALL detected car parts in this frame (orange/amber)
                            for p in detected_parts:
                                self._draw_part(part_vis, p)
                            
                            # 2. Draw damage segmentation mask on top (red)
                            self._draw_damage(part_vis, target.name, dmg.dtype, dmg.conf,
                                              hit["damage_polygon"], (fx1, fy1, fx2, fy2))
                            cv2.putText(part_vis, f"view: {direction}  (frame #{kf.frame_idx})",
                                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                            hit["vis_image"] = part_vis

                        hit["raw_image"] = frame.copy()
                        dir_hits.append(hit)

            # Deduplicate: same (part, damage_type) → keep highest confidence
            deduped = self._dedup_direction_hits(dir_hits)
            report.extend(deduped)

        pbar.close()
        return report

    @staticmethod
    def _dedup_direction_hits(hits: List[Dict], centroid_threshold: float = 0.25) -> List[Dict]:
        """Deduplicate (part, damage_type) pairs from multiple frames of
        the same direction using relative centroid distance.

        Two hits with the same (part, damage_type) are considered the SAME
        physical damage if their relative centroids are within
        `centroid_threshold` (Euclidean distance in 0-1 normalised space).
        In that case, only the highest-confidence entry is kept.

        If centroids are farther apart, they are treated as two physically
        separate damages and BOTH are kept."""
        # Group hits by (part, damage_type)
        from collections import defaultdict
        groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for hit in hits:
            key = (hit["part"], hit["damage_type"])
            groups[key].append(hit)

        result: List[Dict] = []
        for key, group in groups.items():
            # Sort by confidence descending
            group.sort(key=lambda h: h["confidence"], reverse=True)

            # Cluster: each accepted hit is a cluster centre
            clusters: List[Dict] = []
            for hit in group:
                cx, cy = hit.get("_rel_centroid", (0.5, 0.5))
                merged = False
                for existing in clusters:
                    ex, ey = existing.get("_rel_centroid", (0.5, 0.5))
                    dist = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
                    if dist < centroid_threshold:
                        # Same physical damage — existing already has higher conf
                        merged = True
                        break
                if not merged:
                    clusters.append(hit)
            result.extend(clusters)
        return result

    @staticmethod
    def _dedup_cross_view_hits(hits: List[Dict], centroid_threshold: float = 0.50) -> List[Dict]:
        """Deduplicate damage detections across adjacent views for all car parts.

        For singular parts (Front-bumper, Hood, etc.), views like front-right-side and front-left-side
        can share the same part via 'front' (transitively adjacent).
        For side parts (Fender, Doors), adjacent views (e.g. front-right-side and right-side)
        share the exact same physical panel.
        """
        # Build transitive adjacency: views that can "see" the same singular part
        SINGULAR_VIEW_GROUPS = [
            {"front", "front-left-side", "front-right-side"},
            {"back", "back-left-side", "back-right-side"},
        ]

        def views_can_share(v1: str, v2: str, part_name: str) -> bool:
            """True if v1 and v2 see the SAME physical instance of `part_name`."""
            if v1 == v2:
                return True
            
            # If they are directly adjacent, they see the same side panel
            adj = VIEW_ADJACENCY.get(v1, set())
            if v2 in adj:
                return True
                
            # For singular parts, they can be shared across transitive adjacent views
            if part_name in SINGULAR_PARTS:
                for group in SINGULAR_VIEW_GROUPS:
                    if v1 in group and v2 in group:
                        return True
            return False

        kept: List[Dict] = []
        for hit in sorted(hits, key=lambda h: h["confidence"], reverse=True):
            part = hit["part"]
            dtype = hit["damage_type"]
            view = hit["car_view"]
            cx, cy = hit.get("_rel_centroid", (0.5, 0.5))

            is_dup = False
            for k in kept:
                if k["part"] == part and k["damage_type"] == dtype and views_can_share(view, k["car_view"], part):
                    kx, ky = k.get("_rel_centroid", (0.5, 0.5))
                    dist = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
                    if dist < centroid_threshold:
                        is_dup = True
                        break
            
            if not is_dup:
                kept.append(hit)
        return kept

    # ── Main entry point ─────────────────────────────────────────────────

    def run_on_video(self, video_path: str, out_dir: str, out_report_path: Optional[str]) -> List[Dict]:
        """Two-pass pipeline: scan → select keyframes → analyze → report."""

        # Pass 1: fast angle-only scan (stores only metadata, no pixels)
        keyframes = self.scan_video(video_path)
        if not keyframes:
            log.warning("No directions found in video!")
            return []

        # Load pixels for selected keyframes only (memory-efficient)
        self._load_keyframe_pixels(video_path, keyframes)

        # Pass 2: targeted parts + damage analysis on all selected frames
        raw_report = self.analyze_keyframes(keyframes)

        # Cross-view deduplication for singular parts across adjacent views
        report = self._dedup_cross_view_hits(raw_report)

        # Save outputs
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save annotated keyframe image per unique damaged part
        if self.cfg.save_keyframes:
            keyframes_dir = out_path / "keyframes"
            orig_dir = keyframes_dir / "original"
            anno_dir = keyframes_dir / "annotated"
            orig_dir.mkdir(parents=True, exist_ok=True)
            anno_dir.mkdir(parents=True, exist_ok=True)
            
            # Sort report by frame index (ascending) so files are numbered
            # in the order frames appear in the video.
            report.sort(key=lambda h: h.get("frame_index", 0))
            
            used_filenames: Dict[str, int] = {}
            for seq_num, item in enumerate(report, start=1):
                safe_part = item["part"].replace(" ", "_")
                safe_dmg = item["damage_type"].replace(" ", "_")
                safe_view = item["car_view"].replace(" ", "_")
                
                order_prefix = str(seq_num).zfill(3)  # e.g. 001, 002, …
                base_name = f"{order_prefix}_frame{item.get('frame_index', 0)}_{safe_view}_{safe_part}_{safe_dmg}"
                
                # If same base_name already used, add index suffix
                if base_name in used_filenames:
                    used_filenames[base_name] += 1
                    filename = f"{base_name}_{used_filenames[base_name]}.jpg"
                else:
                    used_filenames[base_name] = 1
                    filename = f"{base_name}.jpg"
                
                if "raw_image" in item:
                    cv2.imwrite(str(orig_dir / filename), item["raw_image"])
                    del item["raw_image"]
                
                if "vis_image" in item:
                    cv2.imwrite(str(anno_dir / filename), item["vis_image"])
                    del item["vis_image"]
                
                # Remove internal centroid field before saving
                item.pop("_rel_centroid", None)
                item["image_filename"] = f"keyframes/original/{filename}"
                item["annotated_image_filename"] = f"keyframes/annotated/{filename}"

            log.info("Keyframe images saved -> %s (original & annotated)", keyframes_dir)

        # Generate clean report (without metadata)
        clean_report = []
        for item in report:
            clean_report.append({
                "part": item["part"],
                "damage_type": item["damage_type"],
                "car_view": item["car_view"],
                "confidence": item["confidence"],
                "severity": item["severity"],
                "severity_ratio": item["severity_ratio"]
            })

        import datetime
        severity_rank = {"Minor": 1, "Moderate": 2, "Severe": 3}
        highest_sev = "None"
        if report:
            highest_sev = max((item["severity"] for item in report), key=lambda s: severity_rank.get(s, 0))

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        video_name = Path(video_path).name

        summary_clean = {
            "video_filename": video_name,
            "total_damages_found": len(clean_report),
            "highest_severity": highest_sev,
            "processing_timestamp": now_iso,
            "damages": clean_report
        }

        summary_meta = {
            "video_filename": video_name,
            "total_damages_found": len(report),
            "highest_severity": highest_sev,
            "processing_timestamp": now_iso,
            "damages": report
        }

        # Save standard concise JSON report
        report_path = out_report_path or str(out_path / "damage_report.json")
        Path(report_path).write_text(json.dumps(summary_clean, indent=2))
        log.info("Report saved -> %s", report_path)

        # Save new metadata JSON report
        metadata_path = str(out_path / "damage_metadata.json")
        Path(metadata_path).write_text(json.dumps(summary_meta, indent=2))
        log.info("Metadata saved -> %s", metadata_path)

        return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Two-pass keyframe car damage pipeline")
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--angle-weights", default=str(MODEL_ANGLE_PATH),
                     help=f"Default: {MODEL_ANGLE_PATH}")
    ap.add_argument("--parts-weights", default=str(MODEL_PARTS_PATH),
                     help=f"Default: {MODEL_PARTS_PATH}")
    ap.add_argument("--damage-weights", default=str(MODEL_DAMAGE_PATH),
                     help=f"Default: {MODEL_DAMAGE_PATH}")
    ap.add_argument("--out-dir", default="output",
                     help="Directory for keyframe images and report (default: output/)")
    ap.add_argument("--out-report", default=None,
                     help="Path for JSON report (default: <out-dir>/damage_report.json)")
    ap.add_argument("--parts-conf", type=float, default=0.60)
    ap.add_argument("--damage-conf", type=float, default=0.60)
    ap.add_argument("--crop-strategy", choices=["bbox", "matte"], default="bbox")
    ap.add_argument("--sample-every", type=int, default=1,
                     help="Sample every Nth frame during angle scan (default: 1)")
    ap.add_argument("--min-direction-frames", type=int, default=1,
                     help="Minimum frames that must agree on a direction to confirm it (default: 1)")
    ap.add_argument("--frames-per-direction", type=int, default=7,
                     help="Number of frames to analyze per direction (default: 5)")
    ap.add_argument("--no-draw", action="store_true")
    ap.add_argument("--no-save-keyframes", action="store_true")
    ap.add_argument("--min-sharpness", type=float, default=50.0,
                     help="Min Laplacian variance to accept a frame (0 = disabled, default: 50.0)")
    args = ap.parse_args()

    cfg = PipelineConfig(
        angle_model_path=args.angle_weights,
        parts_model_path=args.parts_weights,
        damage_model_path=args.damage_weights,
        parts_conf=args.parts_conf,
        damage_conf=args.damage_conf,
        crop_strategy=args.crop_strategy,
        sample_every_n=args.sample_every,
        min_direction_frames=args.min_direction_frames,
        frames_per_direction=args.frames_per_direction,
        draw=not args.no_draw,
        save_keyframes=not args.no_save_keyframes,
        min_sharpness=args.min_sharpness,
    )

    pipeline = DamagePipeline(cfg)
    
    # Create a unique subfolder for this video to prevent overwriting
    from pathlib import Path
    video_stem = Path(args.video).stem
    out_dir_for_video = str(Path(args.out_dir) / f"{video_stem}_results")
    
    report = pipeline.run_on_video(args.video, out_dir_for_video, args.out_report)

    print("\n" + "=" * 100)
    print("DAMAGE REPORT")
    print("=" * 100)
    if not report:
        print("No damage detected.")
    for item in report:
        print(f"  {item['part']:<18} {item['damage_type']:<14} view={item['car_view']:<18} "
              f"conf={item['confidence']:.2f}  severity={item['severity']} "
              f"({item['severity_ratio']:.1%})")
    print("=" * 100)


if __name__ == "__main__":
    main()