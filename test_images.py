from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import sys
import os

def get_part_color(part_name: str):
    """Generate a consistent, distinct color for each car part name (BGR format)."""
    colors = [
        (230, 25, 75),   # Red
        (60, 180, 75),   # Green
        (255, 225, 25),  # Yellow
        (0, 130, 200),   # Blue
        (245, 130, 48),  # Orange
        (145, 30, 180),  # Purple
        (70, 240, 240),  # Cyan
        (240, 50, 230),  # Magenta
        (210, 245, 60),  # Lime
        (250, 190, 212), # Pink
        (0, 128, 128),   # Teal
        (220, 190, 255), # Lavender
        (170, 110, 40),  # Brown
        (255, 250, 200), # Beige
        (128, 0, 0),     # Maroon
        (170, 255, 195), # Mint
        (128, 128, 0),   # Olive
        (255, 215, 180), # Coral
        (0, 0, 128),     # Navy
        (128, 128, 128)  # Grey
    ]
    idx = abs(hash(part_name)) % len(colors)
    return colors[idx]

# ── Load models ────────────────────────────────────────────────────────────────
parts_model  = YOLO('models/best_car_part.pt')
damage_model = YOLO('models/best_damage_type.pt')

# ── Per-class confidence thresholds ───────────────────────────────────────────
# Tune these independently for each damage type.
# A detection is only accepted if its confidence >= the threshold set here.
DAMAGE_THRESHOLDS = {
    'dent':         0.50,
    'glass_break':  0.80,
    'scratch':      0.50,
    'smash':        0.80,
    'crack':        0.50,
    'broken_light': 0.50,
    'flat_tire':    0.90,
}

PART_THRESHOLDS = {
    'Back-bumper':     0.50,
    'Back-door':       0.50,
    'Back-wheel':      0.50,
    'Back-window':     0.50,
    'Back-windshield': 0.50,
    'Fender':          0.50,
    'Front-bumper':    0.50,
    'Front-door':      0.50,
    'Front-wheel':     0.50,
    'Front-window':    0.50,
    'Grille':          0.50,
    'Headlight':       0.50,
    'Hood':            0.50,
    'License-plate':   0.50,
    'Mirror':          0.50,
    'Quarter-panel':   0.50,
    'Rocker-panel':    0.50,
    'Roof':            0.50,
    'Tail-light':      0.50,
    'Trunk':           0.50,
    'Windshield':      0.50,
}

# ── Part → allowed damage types ────────────────────────────────────────────────
PART_DAMAGE_MAP = {
    'Back-wheel':          ['flat_tire'],
    'Front-wheel':         ['flat_tire'],

    'Back-window':         ['glass_break', 'crack'],
    'Back-windshield':     ['glass_break', 'crack'],
    'Front-window':        ['glass_break', 'crack'],
    'Windshield':          ['glass_break', 'crack'],

    'Headlight':           ['broken_light', 'crack', 'scratch'],
    'Tail-light':          ['broken_light', 'crack', 'scratch'],

    'Mirror':              ['crack', 'scratch'],

    'Front-bumper':        ['dent', 'smash', 'scratch', 'crack'],
    'Back-bumper':         ['dent', 'smash', 'scratch', 'crack'],

    'Grille':              ['crack', 'scratch'],

    'License-plate':       ['dent', 'scratch', 'smash'],

    'Hood':                ['dent', 'scratch', 'smash', 'crack'],
    'Trunk':               ['dent', 'scratch', 'smash', 'crack'],
    'Roof':                ['dent', 'scratch', 'smash', 'crack'],
    'Fender':              ['dent', 'scratch', 'smash', 'crack'],
    'Front-door':          ['dent', 'scratch', 'smash', 'crack'],
    'Back-door':           ['dent', 'scratch', 'smash', 'crack'],
    'Quarter-panel':       ['dent', 'scratch', 'smash', 'crack'],
    'Rocker-panel':        ['dent', 'scratch', 'smash', 'crack'],
}

_BODY_PANEL_DEFAULT = ['dent', 'scratch', 'smash', 'crack']


def get_allowed_damage(part_name: str) -> list:
    return PART_DAMAGE_MAP.get(part_name, _BODY_PANEL_DEFAULT)


def passes_threshold(damage_class: str, conf: float) -> bool:
    """Check if a damage detection passes its per-class threshold."""
    threshold = DAMAGE_THRESHOLDS.get(damage_class, 0.40)
    return conf >= threshold


def passes_part_threshold(part_name: str, conf: float) -> bool:
    """Check if a part detection passes its per-class threshold."""
    threshold = PART_THRESHOLDS.get(part_name, 0.50)
    return conf >= threshold


# ── Drawing helpers ────────────────────────────────────────────────────────────
def severity_color(conf: float):
    if conf >= 0.70:
        return (0, 0, 220)
    if conf >= 0.40:
        return (0, 140, 255)
    return (0, 200, 100)


def draw_confidence_bar(img, x, y, width, conf, color, bar_h=6):
    cv2.rectangle(img, (x, y), (x + width, y + bar_h), (55, 55, 55), -1)
    fill = int(width * conf)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + bar_h), color, -1)


def draw_label_block(img, x1, y1, part_name: str, damage_rows: list):
    """
    damage_rows: list of (damage_class, conf, passed_threshold)
                 sorted descending by conf.
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fscale    = 0.42
    thickness = 1
    line_h    = 19
    bar_w     = 88
    bar_h     = 6
    lbl_w     = 100
    pct_w     = 55        # wider to fit threshold marker
    pad       = 6
    img_h, img_w = img.shape[:2]

    n_rows  = 1 + len(damage_rows)
    panel_h = n_rows * line_h + pad * 2
    panel_w = pad + lbl_w + bar_w + pct_w + pad

    py1 = max(0, y1 - panel_h)
    px1 = max(0, min(x1, img_w - panel_w))

    cv2.rectangle(img, (px1, py1), (px1 + panel_w, py1 + panel_h),
                  (18, 18, 18), -1)
    cv2.rectangle(img, (px1, py1), (px1 + panel_w, py1 + panel_h),
                  (90, 90, 90), 1)

    # Header: part name
    cv2.putText(img, part_name, (px1 + pad, py1 + pad + 13),
                font, fscale + 0.06, (230, 230, 230), thickness + 1, cv2.LINE_AA)

    # Damage rows
    for i, (d_class, conf, passed) in enumerate(damage_rows):
        ry = py1 + pad + line_h * (i + 1)
        threshold = DAMAGE_THRESHOLDS.get(d_class, 0.40)

        if conf > 0.0 and passed:
            bar_color = severity_color(conf)
            txt_color = (180, 180, 180)
        elif conf > 0.0 and not passed:
            # Detected but below threshold — shown in muted yellow as warning
            bar_color = (0, 180, 200)
            txt_color = (120, 120, 120)
        else:
            bar_color = (55, 55, 55)
            txt_color = (80, 80, 80)

        # Damage class label
        cv2.putText(img, d_class, (px1 + pad, ry + 13),
                    font, fscale, txt_color, thickness, cv2.LINE_AA)

        # Bar
        bx = px1 + pad + lbl_w
        draw_confidence_bar(img, bx, ry + 4, bar_w, conf, bar_color)

        # Draw threshold marker line on the bar
        thresh_x = bx + int(bar_w * threshold)
        cv2.line(img, (thresh_x, ry + 1), (thresh_x, ry + bar_h + 6),
                 (200, 200, 200), 1)

        # Percentage + PASS/FAIL indicator
        if conf > 0.0:
            indicator = "OK" if passed else "LOW"
            ind_color = (0, 200, 100) if passed else (0, 100, 220)
            pct_txt = f"{conf * 100:.1f}%"
            cv2.putText(img, pct_txt, (bx + bar_w + 4, ry + 13),
                        font, fscale, bar_color, thickness, cv2.LINE_AA)
            cv2.putText(img, indicator, (bx + bar_w + 38, ry + 13),
                        font, fscale - 0.04, ind_color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(img, "—", (bx + bar_w + 4, ry + 13),
                        font, fscale, (70, 70, 70), thickness, cv2.LINE_AA)


# ── Main inference ─────────────────────────────────────────────────────────────
def run_inference(image_path: str, output_path: str = 'result.jpg',
                  global_parts_conf: float = 0.10,
                  global_damage_conf: float = 0.10):
    """
    global_parts_conf  / global_damage_conf : passed to YOLO as the minimum
    confidence to even receive a detection from the model. Per-class thresholds
    in DAMAGE_THRESHOLDS / PART_THRESHOLDS are applied on top of these.
    Keep these low (0.10) so per-class thresholds are the real gate.
    """
    print(f"\n{'='*60}")
    print(f"  Car Damage Detection Pipeline")
    print(f"  Image : {image_path}")
    print(f"{'='*60}")
    print(f"\n  Damage thresholds in use:")
    for cls, thr in DAMAGE_THRESHOLDS.items():
        print(f"    {cls:<15} {thr:.2f}")
    print()

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        sys.exit(1)

    # Prepare two separate images
    parts_img = img.copy()
    overlay_parts = parts_img.copy()

    damage_img = img.copy()
    overlay_damage = damage_img.copy()

    drawings = []

    # ── Stage 1: part detection ───────────────────────────────────────────────
    print("[Stage 1] Detecting car parts …")
    parts_results = parts_model.predict(img, conf=global_parts_conf, verbose=False)
    all_boxes = parts_results[0].boxes

    # Filter parts by per-class threshold
    accepted_boxes = []
    for box_idx, box in enumerate(all_boxes):
        part_name = parts_model.names[int(box.cls[0])]
        part_conf = float(box.conf[0])
        if passes_part_threshold(part_name, part_conf):
            accepted_boxes.append((box_idx, box))
        else:
            thr = PART_THRESHOLDS.get(part_name, 0.50)
            print(f"  Skipping {part_name} (conf {part_conf:.2f} < threshold {thr:.2f})")

    if not accepted_boxes:
        print("  No parts passed their thresholds — try lowering PART_THRESHOLDS.\n")
        return

    print(f"  {len(accepted_boxes)}/{len(all_boxes)} part(s) passed thresholds.\n")

    # ── Stage 2: damage detection per part ────────────────────────────────────
    print("[Stage 2] Detecting damage per part …\n")

    for idx, (box_idx, box) in enumerate(accepted_boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        part_cls  = int(box.cls[0])
        part_conf = float(box.conf[0])
        part_name = parts_model.names[part_cls]
        allowed   = get_allowed_damage(part_name)

        print(f"  [{idx+1:02d}] {part_name}  (conf: {part_conf:.2f})")
        print(f"        Allowed damage types : {', '.join(allowed)}")

        # Draw part on parts_img
        part_color = get_part_color(part_name)
        
        # If parts segmentation masks are available, draw them
        if (parts_results[0].masks is not None 
                and box_idx < len(parts_results[0].masks.xy)):
            pts = parts_results[0].masks.xy[box_idx].astype(np.int32)
            if pts.size > 0:
                cv2.fillPoly(overlay_parts, [pts], part_color)
                cv2.polylines(parts_img, [pts], isClosed=True, color=part_color, thickness=2)

        # Draw part name on parts_img

        font = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.42
        thickness = 1
        label = f"{part_name} ({part_conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, font, fscale, thickness)
        label_y = y1 - 6 if y1 - h - 6 > 0 else y1 + h + 6
        cv2.rectangle(parts_img, (x1, label_y - h - 4), (x1 + w + 4, label_y + 4), part_color, -1)
        cv2.putText(parts_img, label, (x1 + 2, label_y), font, fscale, (255, 255, 255), thickness, cv2.LINE_AA)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print("        Crop empty — skipping.\n")
            continue

        dmg_res = damage_model.predict(crop, conf=global_damage_conf, verbose=False)

        # Build raw_map: best conf per damage class from model output
        raw_map = {}
        if len(dmg_res[0].boxes) > 0:
            for d_idx, d_box in enumerate(dmg_res[0].boxes):
                d_cls  = int(d_box.cls[0])
                d_type = damage_model.names[d_cls]
                d_conf = float(d_box.conf[0])
                if d_type not in raw_map or d_conf > raw_map[d_type][0]:
                    raw_map[d_type] = (d_conf, d_idx)

        # Console: raw output
        if raw_map:
            raw_str = ', '.join(
                f"{t}({c:.2f})"
                for t, (c, _) in sorted(raw_map.items(), key=lambda x: -x[1][0])
            )
            print(f"        Raw detections       : {raw_str}")
        else:
            print(f"        Raw detections       : none")

        # Filter: must be allowed AND pass per-class threshold
        conf_map  = {}    # damage_class -> conf  (passed threshold)
        below_map = {}    # damage_class -> conf  (detected but below threshold)

        for d_type in allowed:
            if d_type in raw_map:
                d_conf = raw_map[d_type][0]
                if passes_threshold(d_type, d_conf):
                    conf_map[d_type] = d_conf
                else:
                    below_map[d_type] = d_conf

        if conf_map:
            valid_str = ', '.join(
                f"{t}({c:.2f})" for t, c in
                sorted(conf_map.items(), key=lambda x: -x[1])
            )
            print(f"        Passed threshold     : {valid_str}")
        else:
            print(f"        Passed threshold     : none")

        if below_map:
            below_str = ', '.join(
                f"{t}({c:.2f} < {DAMAGE_THRESHOLDS.get(t,0.40):.2f})"
                for t, c in sorted(below_map.items(), key=lambda x: -x[1])
            )
            print(f"        Below threshold      : {below_str}")

        # Build panel rows — all allowed types, with pass/fail flag
        damage_rows = []
        for d_type in allowed:
            if d_type in conf_map:
                damage_rows.append((d_type, conf_map[d_type], True))
            elif d_type in below_map:
                damage_rows.append((d_type, below_map[d_type], False))
            else:
                damage_rows.append((d_type, 0.0, False))
        damage_rows.sort(key=lambda x: -x[1])

        # Best detection for mask
        best_type   = None
        best_conf   = 0.0
        best_d_idx  = None
        damage_poly = None

        if conf_map:
            best_type  = max(conf_map, key=conf_map.get)
            best_conf  = conf_map[best_type]
            best_d_idx = raw_map[best_type][1]

        if (best_d_idx is not None
                and dmg_res[0].masks is not None
                and best_d_idx < len(dmg_res[0].masks.xy)):
            pts = (dmg_res[0].masks.xy[best_d_idx]
                   + np.array([x1, y1])).astype(np.int32)
            damage_poly = pts

        if best_type:
            status    = f"{best_type} ({best_conf:.2f})"
            box_color = severity_color(best_conf)
        else:
            status    = "no damage detected"
            box_color = (0, 200, 100)

        print(f"        → Final status       : {status}\n")

        drawings.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'part_name':   part_name,
            'damage_rows': damage_rows,
            'damage_poly': damage_poly,
            'box_color':   box_color,
            'best_type':   best_type,
            'best_conf':   best_conf,
        })

    # ── Blend + annotate parts image ──────────────────────────────────────────
    if parts_results[0].masks is not None:
        cv2.addWeighted(overlay_parts, 0.3, parts_img, 0.7, 0, parts_img)

    # ── Blend + annotate damage image ─────────────────────────────────────────
    cv2.addWeighted(overlay_damage, 0.35, damage_img, 0.65, 0, damage_img)

    for d in drawings:
        if d['best_type'] is not None:
            part_color = get_part_color(d['part_name'])
            # Draw bounding box
            cv2.rectangle(damage_img,
                          (d['x1'], d['y1']), (d['x2'], d['y2']),
                          part_color, 2)
            # Draw damage contour
            if d['damage_poly'] is not None:
                cv2.polylines(damage_img, [d['damage_poly']],
                              isClosed=True, color=(0, 0, 255), thickness=2)
            
            # Simple text label: "Part: Damage Type (conf)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fscale = 0.42
            thickness = 1
            label = f"{d['part_name']}: {d['best_type']} ({d['best_conf']:.2f})"
            (w, h), _ = cv2.getTextSize(label, font, fscale, thickness)
            
            # Position label box cleanly above the bounding box
            label_y = d['y1'] - 6 if d['y1'] - h - 6 > 0 else d['y1'] + h + 6
            cv2.rectangle(damage_img, (d['x1'], label_y - h - 4), (d['x1'] + w + 4, label_y + 4), part_color, -1)
            cv2.putText(damage_img, label, (d['x1'] + 2, label_y), font, fscale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Determine filenames
    base, ext = os.path.splitext(output_path)
    parts_path = f"{base}_parts{ext}"
    damage_path = f"{base}_damage{ext}"

    cv2.imwrite(parts_path, parts_img)
    cv2.imwrite(damage_path, damage_img)
    print(f"[Done] Parts detection image saved → {parts_path}")
    print(f"[Done] Damage detection image saved → {damage_path}")

    try:
        max_h = damage_img.shape[0]
        col_width = 450
        items_per_col = max(1, (max_h - 100) // 35)
        num_cols = max(1, (len(drawings) + items_per_col - 1) // items_per_col)
        
        text_img = np.zeros((max_h, col_width * num_cols, 3), dtype=np.uint8)
        
        # Draw Title
        cv2.putText(text_img, "Damage Summary Report", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(text_img, (20, 55), (400, 55), (255, 255, 255), 1)
        
        for i, d in enumerate(drawings):
            col = i // items_per_col
            row = i % items_per_col
            
            x0 = 20 + col * col_width
            y0 = 100 + row * 35
            
            part = d['part_name']
            status = f"{d['best_type']} ({d['best_conf']:.2f})" if d['best_type'] else "No Damage"
            color = (0, 0, 255) if d['best_type'] else (0, 200, 100)
            
            cv2.putText(text_img, f"{part[:18]}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(text_img, status, (x0 + 200, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2 if d['best_type'] else 1, cv2.LINE_AA)

        combined_img = np.hstack((parts_img, damage_img, text_img))
        cv2.namedWindow('Car Parts & Damage Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Car Parts & Damage Detection', combined_img)
        print("Press any key in the window to close …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[Warning] Could not open display window: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Two-stage car damage detection with per-class thresholds',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Per-class thresholds are set inside the script in DAMAGE_THRESHOLDS and
PART_THRESHOLDS dicts — edit those values to tune each class independently.

Examples:
  python test_images.py car8.jpeg
  python test_images.py car8.jpeg --output out.jpg
  python test_images.py car8.jpeg --parts-conf 0.05 --damage-conf 0.05
        """
    )
    ap.add_argument('image',
                    help='Path to input image (e.g. car8.jpeg)')
    ap.add_argument('--output', default='result.jpg',
                    help='Output image path  (default: result.jpg)')
    ap.add_argument('--parts-conf', type=float, default=0.3,
                    help='Global floor for part detections before per-class filter (default: 0.10)')
    ap.add_argument('--damage-conf', type=float, default=0.3,
                    help='Global floor for damage detections before per-class filter (default: 0.10)')
    args = ap.parse_args()

    run_inference(args.image, args.output, args.parts_conf, args.damage_conf)