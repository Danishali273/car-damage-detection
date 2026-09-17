"""
Track damage instances across a 360-degree walkaround video and produce a
full damage assessment output: annotated frames, a damage report, and a
metadata file with full per-detection geometry.

Why this is needed:
    Running detection independently per frame gives you N results for the
    same physical damage as the camera moves past it across a walkaround
    video. Tracking assigns a persistent ID to each damage instance so you
    can collapse all those frames down to one clean result per damage.

Requires:
    pip install -U ultralytics opencv-python

Models used:
    damage_model -> your damage_type_seg_6classes.pt (or similar)
    part_model   -> your car_part / new 29-class part segmentation model

Output layout (all inside OUTPUT_DIR):
    OUTPUT_DIR/
        frames/                     <- one annotated JPG per unique damage
            track{track_id}_{damage_type}_{part}.jpg
        damage_report.json          <- human-facing report: type/part/severity
        metadata.json               <- full geometry + provenance per damage

NOTE ON SEVERITY:
    There is no dedicated severity model in this pipeline yet, so severity
    is estimated heuristically as (damage mask area) / (part mask area) —
    i.e. how much of the affected part the damage covers, not how much of
    the frame. This is roughly invariant to camera distance (a closer shot
    scales both the damage and the part up together), unlike a raw
    frame-area ratio. It's floored per damage type (e.g. "glass shatter"
    and "tire flat" are floored to at least "moderate" since they're
    rarely cosmetic-only). Swap `estimate_severity()` out for a learned
    model later without touching anything else in the pipeline.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

CAR_MODEL_PATH = "models/yolo11n.pt"
DAMAGE_MODEL_PATH = "models/damage_type_seg_6classes.pt"
PART_MODEL_PATH = "models/car_part.pt"          # your new 29-class part model
VIDEO_PATH = r"C:\Users\Zct123\Downloads\car damage detectio v2\testingVideos2\7.mp4"
OUTPUT_DIR = Path("damage_report_" + str(Path(VIDEO_PATH).stem))
FRAMES_DIR = OUTPUT_DIR / "frames"
CONF_THRESHOLD = 0.25
TRACKER_CONFIG = "bytetrack.yaml"

# Severity thresholds, as a fraction of the AFFECTED PART's area covered
# by the damage mask (damage_area / part_area). Tune these against real data.
SEVERITY_THRESHOLDS = {
    "minor": 0.0,      # >= 0% of part area
    "moderate": 0.08,  # >= 8% of part area
    "severe": 0.25,    # >= 25% of part area
}

# Damage types that are functionally serious regardless of visible area
SEVERITY_FLOOR = {
    "glass shatter": "moderate",
    "tire flat": "moderate",
}

# Mapping of which damages are physically possible on which parts
DAMAGE_ALLOWED_ON_PART = {
    'Diggi_Back_Door' : ["dent", "scratch", "crack"],
    'Diggi_Back_Door_Glass': ["glass_break", "glass shatter", "crack"],
    'Front_Bumper': ["dent", "scratch", "crack"],
    'Front_Windshield_Glass': ["glass_break", "glass shatter", "crack"],
    'Grill': ["scratch", "crack", "dent"],
    'Hood_Bonnet': ["dent", "scratch", "crack"],
    'Left_Fender': ["dent", "scratch", "crack"],
    'Left_Front_Door': ["dent", "scratch", "crack"],
    'Left_Front_Door_Glass': ["glass_break", "glass shatter", "crack"],
    'Left_Headlight': ["broken_light", "broken lamp", "glass shatter", "crack", "scratch"],
    'Left_Quarter_Panel': ["dent", "scratch", "crack"],
    'Left_Rear_Door': ["dent", "scratch", "crack"],
    'Left_Rear_Door_Glass': ["glass_break", "glass shatter", "crack"],
    'Left_Running_Board': ["dent", "scratch", "crack"],
    'Left_Side_Mirror': ["dent", "scratch", "crack", "glass_break", "glass shatter", "broken lamp", "broken_light"],
    'Left_Taillight': ["broken_light", "broken lamp", "glass shatter", "crack", "scratch"],
    'Rear_Bumper': ["dent", "scratch", "crack"],
    'Right_Fender': ["dent", "scratch", "crack"],
    'Right_Front_Door': ["dent", "scratch", "crack"],
    'Right_Front_Door_Glass': ["glass_break", "glass shatter", "crack"],
    'Right_Headlight': ["broken_light", "broken lamp", "glass shatter", "crack", "scratch"],
    'Right_Quarter_Panel': ["dent", "scratch", "crack"],
    'Right_Rear_Door': ["dent", "scratch", "crack"],
    'Right_Rear_Door_Glass': ["glass_break", "glass shatter", "crack"],
    'Right_Running_Board': ["dent", "scratch", "crack"],
    'Right_Side_Mirror': ["dent", "scratch", "crack", "glass_break", "glass shatter", "broken lamp", "broken_light"],
    'Right_Taillight': ["broken_light", "broken lamp", "glass shatter", "crack", "scratch"],
    'Roof': ["dent", "scratch", "crack"],
    'tyre': ["tire flat", "flat_tire", "tire_flat"],
    'tire': ["tire flat", "flat_tire", "tire_flat"],
}


def sharpness_score(image_bgr, box):
    """Laplacian variance over the cropped damage region — higher = sharper/less blurred."""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def quality_score(conf, box, image_bgr):
    """Composite score used to pick the single best frame per track_id."""
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1.0)
    sharpness = sharpness_score(image_bgr, box)
    return conf * np.log1p(area) * np.log1p(sharpness)


def mask_area_pixels(mask, box, frame_shape):
    """Area in pixels from a binary mask if available, else from the bbox."""
    h, w = frame_shape[:2]
    if mask is not None:
        m = mask
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return float(m.sum())
    x1, y1, x2, y2 = box
    return max((x2 - x1) * (y2 - y1), 0.0)


def estimate_severity(cls_name, damage_area, part_area):
    """
    Heuristic severity from damage_area / part_area coverage, floored per
    damage type. Returns (level, ratio). If the part area is unknown/zero
    (part detection missed), falls back to "unknown" severity rather than
    guessing.
    """
    if not part_area or part_area <= 0:
        return "unknown", 0.0

    ratio = damage_area / part_area

    level = "minor"
    for name, threshold in sorted(SEVERITY_THRESHOLDS.items(), key=lambda kv: kv[1]):
        if ratio >= threshold:
            level = name

    floor = SEVERITY_FLOOR.get(cls_name.lower())
    if floor:
        order = ["minor", "moderate", "severe"]
        if order.index(floor) > order.index(level):
            level = floor

    return level, ratio


def get_parts_for_mask(part_result, damage_mask, orig_shape):
    """
    Given a part-seg model result on the same frame, find ALL part classes
    that overlap with the damage segmentation mask.
    Returns a list of dicts with the part name plus its own box/mask geometry.
    """
    empty_list = []
    if part_result.masks is None or damage_mask is None:
        return empty_list

    h_orig, w_orig = orig_shape[:2]
    if damage_mask.shape != (h_orig, w_orig):
        dmask = cv2.resize(damage_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    else:
        dmask = damage_mask

    overlapping_parts = []

    for i, (mask, cls_id) in enumerate(zip(part_result.masks.data, part_result.boxes.cls)):
        part_mask = mask.cpu().numpy().astype(np.uint8)
        if part_mask.shape != (h_orig, w_orig):
            part_mask = cv2.resize(part_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        intersection = int((dmask & part_mask).sum())
        if intersection > 0:
            part_area = int(part_mask.sum())
            part_name = part_result.names[int(cls_id)]
            part_box = part_result.boxes.xyxy[i].cpu().numpy().tolist()
            part_poly = part_result.masks.xy[i].tolist() if part_result.masks.xy else None

            overlapping_parts.append({
                "part_name": part_name,
                "part_box": part_box,
                "part_polygon": part_poly,
                "part_area_pixels": part_area,
                "overlap_pixels": intersection,
            })

    # Sort parts by largest overlap first
    overlapping_parts.sort(key=lambda x: x["overlap_pixels"], reverse=True)
    return overlapping_parts


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)
    car_model = YOLO(CAR_MODEL_PATH)
    damage_model = YOLO(DAMAGE_MODEL_PATH)
    part_model = YOLO(PART_MODEL_PATH)

    # track_id -> list of (frame_idx, frame_bgr, box, conf, cls_name, quality, mask, poly)
    track_records = defaultdict(list)

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    print(f"Processing video {VIDEO_PATH}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect Car
        car_results = car_model.predict(frame, verbose=False)[0]
        best_car_box = None
        max_area = 0
        for box in car_results.boxes:
            if int(box.cls[0]) in [2, 5, 7]:  # car, bus, truck (COCO classes)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_car_box = (x1, y1, x2, y2)

        if best_car_box is None:
            frame_idx += 1
            continue

        # Crop to the vehicle with small padding
        cx1, cy1, cx2, cy2 = best_car_box
        h, w = frame.shape[:2]
        pad = 20
        cx1, cy1 = max(0, cx1 - pad), max(0, cy1 - pad)
        cx2, cy2 = min(w, cx2 + pad), min(h, cy2 + pad)
        car_crop = frame[cy1:cy2, cx1:cx2]

        if car_crop.size == 0:
            frame_idx += 1
            continue

        # 2. Track Damages on the Crop
        results = damage_model.track(
            source=car_crop,
            conf=CONF_THRESHOLD,
            tracker=TRACKER_CONFIG,
            persist=True,
            verbose=False,
        )
        result = results[0]

        frame_bgr = result.orig_img
        has_tracks = (result.boxes is not None and result.boxes.id is not None)
        has_masks = (result.masks is not None)
        if has_tracks:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()
            if has_masks:
                masks_data = result.masks.data.cpu().numpy().astype(np.uint8)
                masks_poly = result.masks.xy
            else:
                masks_data = [None] * len(boxes)
                masks_poly = [None] * len(boxes)

            for box, conf, cls_id, track_id, dmask, dpoly in zip(
                boxes, confs, clses, track_ids, masks_data, masks_poly
            ):
                cls_name = result.names[int(cls_id)]
                q = quality_score(float(conf), box, frame_bgr)
                track_records[int(track_id)].append(
                    (frame_idx, frame_bgr.copy(), box, float(conf), cls_name, q, dmask, dpoly)
                )
        
        frame_idx += 1
        print(f"\rProcessed frame {frame_idx}", end="")

    print(f"\nFound {len(track_records)} unique damage tracks across {frame_idx} frames.")

    damage_report = []   # human-facing: type, part, severity
    metadata = []         # full geometry + provenance

    # 1. Evaluate and filter tracks
    processed_tracks = []
    for track_id, records in track_records.items():
        best_frame_idx, best_frame, best_box, best_conf, cls_name, q, best_mask, best_poly = max(
            records, key=lambda r: r[5]
        )

        part_result = part_model.predict(best_frame, verbose=False)[0]
        overlapping_parts = get_parts_for_mask(part_result, best_mask, best_frame.shape)
        damage_area = mask_area_pixels(best_mask, best_box, best_frame.shape)

        for part_info in overlapping_parts:
            part_name = part_info["part_name"]
            
            # Filter out damages on the background/ground
            if part_name == "unknown":
                continue
                
            # Filter out impossible damage combinations (e.g. lamp broken on a windshield)
            if part_name in DAMAGE_ALLOWED_ON_PART:
                allowed_damages = DAMAGE_ALLOWED_ON_PART[part_name]
                # Clean up strings for loose matching
                clean_cls = cls_name.lower().replace("_", " ")
                is_allowed = False
                for ad in allowed_damages:
                    clean_ad = ad.lower().replace("_", " ")
                    if clean_cls in clean_ad or clean_ad in clean_cls:
                        is_allowed = True
                        break
                if not is_allowed:
                    continue
            
            # Filter out damages that barely touch a part (e.g., less than 10% of damage area is on this part)
            if damage_area > 0 and part_info["overlap_pixels"] > 0:
                overlap_ratio = part_info["overlap_pixels"] / damage_area
                if overlap_ratio < 0.1:
                    continue

            # Make damage area strictly dependent on the part: use the intersection area
            if part_info["overlap_pixels"] > 0:
                effective_damage_area = part_info["overlap_pixels"]
            else:
                effective_damage_area = damage_area

            severity, area_ratio = estimate_severity(
                cls_name, effective_damage_area, part_info["part_area_pixels"]
            )

            processed_tracks.append({
                "track_id": track_id,
                "records": records,
                "best_frame_idx": best_frame_idx,
                "best_frame": best_frame,
                "best_box": best_box,
                "best_conf": best_conf,
                "cls_name": cls_name,
                "best_poly": best_poly,
                "part_info": part_info,
                "damage_area": effective_damage_area,
                "severity": severity,
                "area_ratio": area_ratio,
                "q": q
            })

    # 2. Deduplicate tracks (same damage type, same part, non-overlapping in time)
    final_tracks = []
    grouped_tracks = defaultdict(list)
    for pt in processed_tracks:
        grouped_tracks[(pt["cls_name"], pt["part_info"]["part_name"])].append(pt)

    for (g_cls_name, g_part_name), tracks_in_group in grouped_tracks.items():
        # Sort by quality so we keep the best representation
        tracks_in_group.sort(key=lambda x: x["q"], reverse=True)
        merged_groups = []
        
        for t in tracks_in_group:
            placed = False
            t_frames = {r[0] for r in t["records"]}
            
            for mg in merged_groups:
                mg_frames = set().union(*[{r[0] for r in mt["records"]} for mt in mg])
                
                # Spatial overlap check against the best track in this group
                best_mg = mg[0]
                box_t = t["best_box"]
                box_mg = best_mg["best_box"]
                
                ix1 = max(box_t[0], box_mg[0])
                iy1 = max(box_t[1], box_mg[1])
                ix2 = min(box_t[2], box_mg[2])
                iy2 = min(box_t[3], box_mg[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                
                area_t = (box_t[2] - box_t[0]) * (box_t[3] - box_t[1])
                area_mg = (box_mg[2] - box_mg[0]) * (box_mg[3] - box_mg[1])
                iou = inter / (area_t + area_mg - inter) if (area_t + area_mg - inter) > 0 else 0
                
                # Merge if they don't overlap in time (tracker lost ID) 
                # OR if they spatially overlap heavily (duplicate simultaneous boxes)
                if len(t_frames.intersection(mg_frames)) == 0 or iou > 0.15:
                    mg.append(t)
                    placed = True
                    break
            
            if not placed:
                merged_groups.append([t])
                
        # Pick the best track from each merged group
        for mg in merged_groups:
            best_track = mg[0]
            combined_records = []
            for mt in mg:
                combined_records.extend(mt["records"])
            best_track["records"] = combined_records
            final_tracks.append(best_track)

    for pt in final_tracks:
        track_id = pt["track_id"]
        records = pt["records"]
        best_frame_idx = pt["best_frame_idx"]
        best_frame = pt["best_frame"]
        best_box = pt["best_box"]
        best_conf = pt["best_conf"]
        cls_name = pt["cls_name"]
        best_poly = pt["best_poly"]
        part_info = pt["part_info"]
        damage_area = pt["damage_area"]
        severity = pt["severity"]
        area_ratio = pt["area_ratio"]

        out_filename = f"track{track_id}_{cls_name}_{part_info['part_name']}.jpg"
        out_path = FRAMES_DIR / out_filename
        annotated = best_frame.copy()

        # Draw Part Polygon/Box (Green)
        part_poly = part_info.get("part_polygon")
        part_box = part_info.get("part_box")
        if part_poly is not None and len(part_poly) > 0:
            part_poly_pts = [np.int32(part_poly)]
            overlay = annotated.copy()
            cv2.fillPoly(overlay, part_poly_pts, (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
            cv2.polylines(annotated, part_poly_pts, isClosed=True, color=(0, 255, 0), thickness=2)
        elif part_box is not None:
            px1, py1, px2, py2 = map(int, part_box)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)

        # Draw Damage Polygon/Box (Red)
        if best_poly is not None and len(best_poly) > 0:
            poly_pts = [np.int32(best_poly)]
            overlay = annotated.copy()
            cv2.fillPoly(overlay, poly_pts, (0, 0, 255))
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
            cv2.polylines(annotated, poly_pts, isClosed=True, color=(0, 0, 255), thickness=2)
        else:
            x1, y1, x2, y2 = map(int, best_box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        x1, y1 = int(best_box[0]), int(best_box[1])
        cv2.putText(
            annotated,
            f"{cls_name} on {part_info['part_name']} [{severity}] ({best_conf:.2f})",
            (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
        cv2.imwrite(str(out_path), annotated)

        damage_report.append({
            "track_id": track_id,
            "damage_type": cls_name,
            "part": part_info["part_name"],
            "severity": severity,
            "confidence": round(best_conf, 3),
            "frame_image": f"frames/{out_filename}",
        })

        metadata.append({
            "track_id": track_id,
            "damage_type": cls_name,
            "confidence": round(best_conf, 3),
            "severity": severity,
            "severity_ratio_damage_over_part": round(area_ratio, 5),
            "num_frames_seen": len(records),
            "best_frame_index": best_frame_idx,
            "frame_width": int(best_frame.shape[1]),
            "frame_height": int(best_frame.shape[0]),
            "damage_box_xyxy": [float(v) for v in best_box],
            "damage_polygon": best_poly.tolist() if best_poly is not None else None,
            "damage_area_pixels": damage_area,
            "part": {
                "name": part_info["part_name"],
                "box_xyxy": part_info["part_box"],
                "polygon": part_info["part_polygon"],
                "area_pixels": part_info["part_area_pixels"],
                "mask_overlap_pixels": part_info["overlap_pixels"],
            },
            "all_seen_frame_indices": [r[0] for r in records],
            "frame_image": f"frames/{out_filename}",
        })

    with open(OUTPUT_DIR / "damage_report.json", "w") as f:
        json.dump({
            "video": str(VIDEO_PATH),
            "total_frames_processed": frame_idx,
            "total_damages_found": len(damage_report),
            "damages": damage_report,
        }, f, indent=2)

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump({
            "video": str(VIDEO_PATH),
            "damage_model": DAMAGE_MODEL_PATH,
            "part_model": PART_MODEL_PATH,
            "conf_threshold": CONF_THRESHOLD,
            "tracker": TRACKER_CONFIG,
            "total_frames_processed": frame_idx,
            "detections": metadata,
        }, f, indent=2)

    print(f"Wrote {len(damage_report)} damages to {OUTPUT_DIR}/damage_report.json")
    print(f"Wrote metadata to {OUTPUT_DIR}/metadata.json")
    print(f"Annotated frames saved to {FRAMES_DIR}/")

    return damage_report, metadata


if __name__ == "__main__":
    main()
