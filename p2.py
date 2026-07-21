"""
damage_pipeline.py — Three-Model Car Damage Detection over Video
===================================================================
Chains three independently-trained YOLOv8 models on every processed frame:

    1) Angle model      (classify)  -> which side of the car the camera sees
    2) Parts model       (segment)   -> where each panel/part is, cropped
    3) Damage model       (segment)   -> damage masks (not boxes) inside each crop

Design notes (why this isn't a line-for-line copy of the original)
--------------------------------------------------------------------
- One `PipelineConfig` dataclass holds every tunable instead of a dozen
  module-level constants — makes it trivial to run multiple configurations
  (e.g. grid-searching conf thresholds) without editing the file.
- `CropStrategy` is a first-class, swappable setting. The earlier diagnosis
  in this conversation was that the damage model sees clean, tightly-boxed
  training crops but noisier, background-bleeding crops in production. So
  this version supports three crop modes (bbox / padded-bbox / mask-matted)
  instead of hard-coding a single bbox crop, so you can A/B which one your
  damage model actually agrees with.
- Voting/aggregation is one small `EvidenceLedger` class using plain
  Counters + running stats, instead of a PartRecord/DamageInstance class
  pair. Functionally equivalent (confirm damage only after enough frames
  vote for it) but far less code to maintain.
- Part re-attribution (deciding which part a damage box "really" belongs
  to when boxes overlap) is intentionally simplified to nearest-centroid
  containment only — the original's extra fallback branches are collapsed
  into one helper.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("damage_pipeline")

BASE_DIR = Path(__file__).resolve().parent

# ── Model weights — edit these paths for your environment ───────────────────
MODEL_ANGLE_PATH  = BASE_DIR / "models" / "best_car_angle.pt"
MODEL_PARTS_PATH  = BASE_DIR / "models" / "best_car_part.pt"
MODEL_DAMAGE_PATH = BASE_DIR / "models" / "best_damage_type.pt"


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    angle_model_path: str = str(MODEL_ANGLE_PATH)
    parts_model_path: str = str(MODEL_PARTS_PATH)
    damage_model_path: str = str(MODEL_DAMAGE_PATH)

    parts_conf: float = 0.50
    damage_conf: float = 0.60

    # How crops are built before being handed to the damage model.
    #   "bbox"   -> raw axis-aligned crop (original behaviour)
    #   "padded" -> bbox crop + a margin, matches loosely-annotated training data better
    #   "matte"  -> bbox crop with everything outside the part mask blacked out,
    #               forces the damage model to focus on the panel and ignore
    #               whatever the parts model over-included in the box
    crop_strategy: str = "matte"
    crop_padding_ratio: float = 0.08  # only used by "padded"

    # A frame's direction is only trusted once it's been the majority vote
    # over this many recent frames — smooths single-frame misclassifications.
    direction_smoothing_window: int = 7

    # Confirmation thresholds for the evidence ledger (see EvidenceLedger).
    min_votes: int = 4
    min_vote_ratio: float = 0.20

    # Same-instance clustering radius in normalized crop-centroid distance.
    cluster_radius: float = 0.28

    frame_skip: int = 1
    draw: bool = True


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
    "back": ["Back-bumper", "Trunk", "Tail-light", "Back-windshield"],
    "left-side": ["Front-door", "Back-door", "Front-wheel", "Back-wheel", "Fender", "Quarter-panel", "Mirror", "Rocker-panel"],
    "right-side": ["Front-door", "Back-door", "Front-wheel", "Back-wheel", "Fender", "Quarter-panel", "Mirror", "Rocker-panel"],
    "front-left-side": ["Front-bumper", "Fender", "Mirror", "Headlight", "Windshield"],
    "front-right-side": ["Front-bumper", "Fender", "Mirror", "Headlight", "Windshield"],
    "back-left-side": ["Back-bumper", "Quarter-panel", "Tail-light", "Back-windshield"],
    "back-right-side": ["Back-bumper", "Quarter-panel", "Tail-light", "Back-windshield"],
}

DAMAGE_ALLOWED_ON_PART: Dict[str, List[str]] = {
    "Front-wheel": ["flat_tire"], "Back-wheel": ["flat_tire"],
    "Windshield": ["glass_break"], "Back-windshield": ["glass_break"],
    "Headlight": ["broken_light"], "Tail-light": ["broken_light"],
    "Mirror": ["crack", "scratch"],
    "Front-bumper": ["dent", "scratch", "crack"], "Back-bumper": ["dent", "scratch", "crack"],
    "Hood": ["dent", "scratch", "crack"], "Trunk": ["dent", "scratch", "crack"],
    "Fender": ["dent", "scratch", "crack"], "Front-door": ["dent", "scratch", "crack"],
    "Back-door": ["dent", "scratch", "crack"], "Quarter-panel": ["dent", "scratch", "crack"],
    "Rocker-panel": ["dent", "scratch", "crack"],
}

SEVERITY_BANDS: List[Tuple[float, str]] = [(0.05, "Minor"), (0.20, "Moderate"), (1.01, "Severe")]


def severity_for(ratio: float) -> str:
    for cutoff, label in SEVERITY_BANDS:
        if ratio <= cutoff:
            return label
    return "Severe"


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

    if cfg.crop_strategy == "padded":
        pad_x = int((x2 - x1) * cfg.crop_padding_ratio)
        pad_y = int((y2 - y1) * cfg.crop_padding_ratio)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        return frame[y1:y2, x1:x2], (x1, y1)

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
# EVIDENCE LEDGER — temporal voting, in far fewer lines than the original
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _Evidence:
    votes: int = 0
    frames: set = field(default_factory=set)
    best_conf: float = 0.0
    best_view: str = ""
    cx_sum: float = 0.0
    cy_sum: float = 0.0
    max_damage_area: float = 0.0
    max_part_area: float = 0.0

    def add(self, frame_idx: int, conf: float, cx: float, cy: float, damage_area: float, part_area: float, view: str) -> None:
        if frame_idx in self.frames:
            if conf > self.best_conf:
                self.best_conf = conf
                self.best_view = view
            return
        self.frames.add(frame_idx)
        self.votes += 1
        if conf > self.best_conf or self.best_conf == 0.0:
            self.best_conf = conf
            self.best_view = view
        self.cx_sum += cx
        self.cy_sum += cy
        self.max_damage_area = max(self.max_damage_area, damage_area)
        self.max_part_area = max(self.max_part_area, part_area)

    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.cx_sum / self.votes, self.cy_sum / self.votes) if self.votes else (0.5, 0.5)

    @property
    def severity_ratio(self) -> float:
        return (self.max_damage_area / self.max_part_area) if self.max_part_area > 0 else 0.0


class EvidenceLedger:
    """Accumulates (part, damage_type) evidence across frames and confirms
    only what survives a minimum-vote / minimum-ratio bar."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._entries: Dict[Tuple[str, str], List[_Evidence]] = defaultdict(list)
        self._part_frames_seen: Counter = Counter()

    def note_part_seen(self, part_name: str, frame_idx: int) -> None:
        self._part_frames_seen[part_name] += 1

    def note_damage(self, part_name: str, dtype: str, frame_idx: int, conf: float,
                     cx: float, cy: float, damage_area: float, part_area: float, view: str) -> None:
        key = (part_name, dtype)
        bucket = self._entries[key]
        best, best_dist = None, float("inf")
        for ev in bucket:
            ecx, ecy = ev.centroid
            d = ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5
            if d < best_dist:
                best, best_dist = ev, d
        if best is None or best_dist > self.cfg.cluster_radius:
            best = _Evidence()
            bucket.append(best)
        best.add(frame_idx, conf, cx, cy, damage_area, part_area, view)

    def confirmed_report(self) -> List[Dict]:
        report = []
        for (part_name, dtype), bucket in self._entries.items():
            seen = max(self._part_frames_seen[part_name], 1)
            for ev in bucket:
                ratio = ev.votes / seen
                if ev.votes >= self.cfg.min_votes and ratio >= self.cfg.min_vote_ratio:
                    report.append({
                        "part": part_name,
                        "damage_type": dtype,
                        "car_view": ev.best_view,
                        "confidence": round(ev.best_conf, 3),
                        "severity": severity_for(ev.severity_ratio),
                        "severity_ratio": round(ev.severity_ratio, 4),
                    })
        return sorted(report, key=lambda r: (-r["confidence"]))


# ═══════════════════════════════════════════════════════════════════════════
# DIRECTION SMOOTHING — majority vote over a rolling window
# ═══════════════════════════════════════════════════════════════════════════

class DirectionSmoother:
    def __init__(self, window: int):
        self._buf: List[str] = []
        self._window = max(1, window)

    def push(self, direction: str) -> str:
        self._buf.append(direction)
        self._buf = self._buf[-self._window:]
        return Counter(self._buf).most_common(1)[0][0]


# ═══════════════════════════════════════════════════════════════════════════
# PART RE-ATTRIBUTION — which part does a damage box really belong to?
# ═══════════════════════════════════════════════════════════════════════════

def reattribute(damage_center: Tuple[int, int], parts: List[PartBox], fallback: PartBox) -> PartBox:
    """Pick whichever part's mask actually contains the damage centroid.
    Falls back to the crop's own part when no mask contains it or the
    damage type wouldn't be valid there anyway."""
    best, best_score = None, -1e9
    for p in parts:
        if p.mask_xy is None or p.mask_xy.size < 6:
            continue
        score = cv2.pointPolygonTest(p.mask_xy, damage_center, True)
        if score > best_score:
            best, best_score = p, score
    if best is not None and best_score >= -15.0:
        return best
    return fallback


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class DamagePipeline:
    def __init__(self, cfg: PipelineConfig):
        log.info("Loading angle / parts / damage models …")
        self.cfg = cfg
        self.angle = AngleStage(cfg.angle_model_path)
        self.parts = PartsStage(cfg.parts_model_path)
        self.damage = DamageStage(cfg.damage_model_path)
        self.direction_smoother = DirectionSmoother(cfg.direction_smoothing_window)
        self.ledger = EvidenceLedger(cfg)
        log.info("Models ready.")

    def _draw_damage(self, vis: np.ndarray, dmg: DamageBox, origin: Tuple[int, int],
                      fallback_box: Tuple[int, int, int, int], part_name: str) -> None:
        """Render the damage as a filled, semi-transparent segmentation mask
        (this is a segmentation model — draw what it actually predicted,
        not a bounding box around it). Falls back to an outline box only
        when the model returned no polygon for this detection."""
        ox, oy = origin
        if dmg.mask_in_crop is not None and dmg.mask_in_crop.size >= 6:
            poly = (dmg.mask_in_crop + np.array([ox, oy])).astype(np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [poly], (0, 0, 255))
            cv2.polylines(vis, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.addWeighted(overlay, 0.40, vis, 0.60, 0, vis)
            label_x, label_y = poly[:, 0].min(), poly[:, 1].min()
        else:
            dx1, dy1, dx2, dy2 = fallback_box
            cv2.rectangle(vis, (ox + dx1, oy + dy1), (ox + dx2, oy + dy2), (0, 0, 255), 2)
            label_x, label_y = ox + dx1, oy + dy1

        cv2.putText(vis, f"{part_name}:{dmg.dtype} {dmg.conf:.2f}",
                    (int(label_x), max(15, int(label_y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        raw_view, _ = self.angle.infer(frame)
        car_direction = CAMERA_TO_CAR_DIRECTION.get(raw_view, raw_view)
        stable_direction = self.direction_smoother.push(car_direction)

        allowed_parts = PARTS_VISIBLE_FROM.get(stable_direction, [])
        detected_parts = self.parts.infer(frame, allowed_parts, self.cfg.parts_conf)

        vis = frame.copy()
        for part in detected_parts:
            self.ledger.note_part_seen(part.name, frame_idx)
            crop, (ox, oy) = build_crop(frame, part, self.cfg)
            if crop.size == 0:
                continue

            allowed_dmg = DAMAGE_ALLOWED_ON_PART.get(part.name, [])
            for dmg in self.damage.infer(crop, allowed_dmg, self.cfg.damage_conf):
                dx1, dy1, dx2, dy2 = dmg.xyxy_in_crop
                frame_center = (ox + (dx1 + dx2) // 2, oy + (dy1 + dy2) // 2)

                target = reattribute(frame_center, detected_parts, fallback=part)
                if dmg.dtype not in DAMAGE_ALLOWED_ON_PART.get(target.name, []):
                    target = part  # re-attribution guard: revert if physically invalid there

                crop_h, crop_w = crop.shape[:2]
                cx = ((dx1 + dx2) / 2) / crop_w
                cy = ((dy1 + dy2) / 2) / crop_h
                if dmg.mask_in_crop is not None and dmg.mask_in_crop.size >= 6:
                    damage_area = float(cv2.contourArea(dmg.mask_in_crop.astype(np.float32)))
                else:
                    damage_area = float((dx2 - dx1) * (dy2 - dy1))

                self.ledger.note_damage(
                    part_name=target.name, dtype=dmg.dtype, frame_idx=frame_idx,
                    conf=dmg.conf, cx=cx, cy=cy,
                    damage_area=float(damage_area), part_area=target.area_px,
                    view=stable_direction,
                )

                if self.cfg.draw:
                    self._draw_damage(vis, dmg, origin=(ox, oy), fallback_box=(dx1, dy1, dx2, dy2),
                                       part_name=target.name)

            if self.cfg.draw:
                x1, y1, x2, y2 = part.xyxy
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 180, 0), 1)
                cv2.putText(vis, part.name, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1)

        if self.cfg.draw:
            cv2.putText(vis, f"view: {stable_direction}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return vis

    def run_on_video(self, video_path: str, out_video_path: Optional[str], out_report_path: Optional[str]) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        writer = None
        if out_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_video_path, fourcc, fps / max(1, self.cfg.frame_skip), (w, h))

        frame_idx = 0
        processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % self.cfg.frame_skip != 0:
                continue
            annotated = self.process_frame(frame, frame_idx)
            processed += 1
            if writer is not None:
                writer.write(annotated)
            
            if total_frames > 0:
                print(f"\r[INFO] Processing frame {frame_idx}/{total_frames} ({(frame_idx / total_frames)*100:.1f}%)", end="", flush=True)
            else:
                print(f"\r[INFO] Processing frame {frame_idx}...", end="", flush=True)

        print()  # Add a newline after the progress indicator finishes
        cap.release()
        if writer is not None:
            writer.release()
        log.info("Processed %d frames (of %d total).", processed, frame_idx)

        report = self.ledger.confirmed_report()
        if out_report_path:
            Path(out_report_path).write_text(json.dumps(report, indent=2))
            log.info("Report saved -> %s", out_report_path)
        return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Three-model car damage pipeline (video)")
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--angle-weights", default=str(MODEL_ANGLE_PATH),
                     help=f"Default: {MODEL_ANGLE_PATH}")
    ap.add_argument("--parts-weights", default=str(MODEL_PARTS_PATH),
                     help=f"Default: {MODEL_PARTS_PATH}")
    ap.add_argument("--damage-weights", default=str(MODEL_DAMAGE_PATH),
                     help=f"Default: {MODEL_DAMAGE_PATH}")
    ap.add_argument("--out-video", default="annotated_output.mp4")
    ap.add_argument("--out-report", default="damage_report.json")
    ap.add_argument("--parts-conf", type=float, default=0.50)
    ap.add_argument("--damage-conf", type=float, default=0.60)
    ap.add_argument("--crop-strategy", choices=["bbox", "padded", "matte"], default="bbox")
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--min-votes", type=int, default=4)
    ap.add_argument("--min-vote-ratio", type=float, default=0.20)
    ap.add_argument("--no-draw", action="store_true")
    args = ap.parse_args()

    cfg = PipelineConfig(
        angle_model_path=args.angle_weights,
        parts_model_path=args.parts_weights,
        damage_model_path=args.damage_weights,
        parts_conf=args.parts_conf,
        damage_conf=args.damage_conf,
        crop_strategy=args.crop_strategy,
        frame_skip=args.frame_skip,
        min_votes=args.min_votes,
        min_vote_ratio=args.min_vote_ratio,
        draw=not args.no_draw,
    )

    pipeline = DamagePipeline(cfg)
    report = pipeline.run_on_video(args.video, args.out_video, args.out_report)

    print("\n" + "=" * 100)
    print("CONFIRMED DAMAGE REPORT")
    print("=" * 100)
    if not report:
        print("No confirmed damage.")
    for item in report:
        print(f"  {item['part']:<18} {item['damage_type']:<14} view={item['car_view']:<18} "
              f"conf={item['confidence']:.2f}  severity={item['severity']} "
              f"({item['severity_ratio']:.1%})")
    print("=" * 100)


if __name__ == "__main__":
    main()