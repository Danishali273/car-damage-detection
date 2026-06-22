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
        (128, 128, 128), # Grey
    ]
    idx = abs(hash(part_name)) % len(colors)
    return colors[idx]


# ── Load models ────────────────────────────────────────────────────────────────
parts_model  = YOLO('models/best_car_part.pt')
damage_model = YOLO('models/best_damage_type.pt')

# ── Per-class confidence thresholds ───────────────────────────────────────────
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
    return conf >= DAMAGE_THRESHOLDS.get(damage_class, 0.40)


def passes_part_threshold(part_name: str, conf: float) -> bool:
    return conf >= PART_THRESHOLDS.get(part_name, 0.50)


def severity_color(conf: float):
    if conf >= 0.70:
        return (0, 0, 220)
    if conf >= 0.40:
        return (0, 140, 255)
    return (0, 200, 100)


# ── Per-frame inference ────────────────────────────────────────────────────────
def process_frame(frame: np.ndarray,
                  global_parts_conf: float,
                  global_damage_conf: float):
    """
    Run the two-stage pipeline on a single frame.
    Returns (parts_frame, damage_frame) — both annotated copies.
    """
    parts_frame  = frame.copy()
    overlay_p    = parts_frame.copy()
    damage_frame = frame.copy()
    overlay_d    = damage_frame.copy()

    # ── Stage 1: part detection ───────────────────────────────────────────────
    parts_results = parts_model.predict(frame, conf=global_parts_conf, verbose=False)
    all_boxes = parts_results[0].boxes

    accepted_boxes = []
    for box_idx, box in enumerate(all_boxes):
        part_name = parts_model.names[int(box.cls[0])]
        part_conf = float(box.conf[0])
        if passes_part_threshold(part_name, part_conf):
            accepted_boxes.append((box_idx, box))

    # ── Stage 2: damage per part ──────────────────────────────────────────────
    drawings = []
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fscale    = 0.42
    thickness = 1

    for idx, (box_idx, box) in enumerate(accepted_boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        part_cls  = int(box.cls[0])
        part_conf = float(box.conf[0])
        part_name = parts_model.names[part_cls]
        allowed   = get_allowed_damage(part_name)
        part_color = get_part_color(part_name)

        # Draw part segmentation mask on parts frame
        if (parts_results[0].masks is not None
                and box_idx < len(parts_results[0].masks.xy)):
            pts = parts_results[0].masks.xy[box_idx].astype(np.int32)
            if pts.size > 0:
                cv2.fillPoly(overlay_p, [pts], part_color)
                cv2.polylines(parts_frame, [pts], isClosed=True,
                              color=part_color, thickness=2)

        # Part label on parts frame
        label = f"{part_name} ({part_conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, font, fscale, thickness)
        label_y = y1 - 6 if y1 - h - 6 > 0 else y1 + h + 6
        cv2.rectangle(parts_frame,
                      (x1, label_y - h - 4), (x1 + w + 4, label_y + 4),
                      part_color, -1)
        cv2.putText(parts_frame, label, (x1 + 2, label_y),
                    font, fscale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Crop and run damage model
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        dmg_res = damage_model.predict(crop, conf=global_damage_conf, verbose=False)

        raw_map = {}
        if len(dmg_res[0].boxes) > 0:
            for d_box in dmg_res[0].boxes:
                d_cls  = int(d_box.cls[0])
                d_type = damage_model.names[d_cls]
                d_conf = float(d_box.conf[0])
                if d_type not in raw_map or d_conf > raw_map[d_type][0]:
                    raw_map[d_type] = (d_conf, int(d_box.cls[0]))

        conf_map  = {}
        below_map = {}
        for d_type in allowed:
            if d_type in raw_map:
                d_conf = raw_map[d_type][0]
                if passes_threshold(d_type, d_conf):
                    conf_map[d_type] = d_conf
                else:
                    below_map[d_type] = d_conf

        # Best damage for this part
        best_type  = None
        best_conf  = 0.0
        best_d_idx = None
        damage_poly = None

        if conf_map:
            best_type  = max(conf_map, key=conf_map.get)
            best_conf  = conf_map[best_type]
            # Re-find box index by matching class name
            for d_idx_inner, d_box in enumerate(dmg_res[0].boxes):
                if damage_model.names[int(d_box.cls[0])] == best_type:
                    best_d_idx = d_idx_inner
                    break

        if (best_d_idx is not None
                and dmg_res[0].masks is not None
                and best_d_idx < len(dmg_res[0].masks.xy)):
            pts = (dmg_res[0].masks.xy[best_d_idx]
                   + np.array([x1, y1])).astype(np.int32)
            damage_poly = pts

        drawings.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'part_name':  part_name,
            'part_color': part_color,
            'damage_poly': damage_poly,
            'best_type':  best_type,
            'best_conf':  best_conf,
        })

    # Blend parts overlay
    if parts_results[0].masks is not None:
        cv2.addWeighted(overlay_p, 0.3, parts_frame, 0.7, 0, parts_frame)

    # Annotate damage frame
    cv2.addWeighted(overlay_d, 0.35, damage_frame, 0.65, 0, damage_frame)

    for d in drawings:
        if d['best_type'] is not None:
            cv2.rectangle(damage_frame,
                          (d['x1'], d['y1']), (d['x2'], d['y2']),
                          d['part_color'], 2)
            if d['damage_poly'] is not None:
                cv2.polylines(damage_frame, [d['damage_poly']],
                              isClosed=True, color=(0, 0, 255), thickness=2)

            label = f"{d['part_name']}: {d['best_type']} ({d['best_conf']:.2f})"
            (w, h), _ = cv2.getTextSize(label, font, fscale, thickness)
            label_y = d['y1'] - 6 if d['y1'] - h - 6 > 0 else d['y1'] + h + 6
            cv2.rectangle(damage_frame,
                          (d['x1'], label_y - h - 4),
                          (d['x1'] + w + 4, label_y + 4),
                          d['part_color'], -1)
            cv2.putText(damage_frame, label, (d['x1'] + 2, label_y),
                        font, fscale, (255, 255, 255), thickness, cv2.LINE_AA)

    return parts_frame, damage_frame


# ── HUD overlay ───────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, frame_idx: int, fps: float,
             total_frames: int, mode: str):
    """Burn a small status bar onto the frame."""
    h, w = frame.shape[:2]
    pct = frame_idx / max(total_frames, 1) * 100
    txt = (f"Frame {frame_idx}/{total_frames} ({pct:.1f}%)  "
           f"FPS:{fps:.1f}  [{mode}]")
    cv2.rectangle(frame, (0, h - 22), (w, h), (18, 18, 18), -1)
    cv2.putText(frame, txt, (8, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# ── Main video pipeline ────────────────────────────────────────────────────────
def run_video(video_path: str,
              output_path: str   = 'result_video.mp4',
              global_parts_conf: float = 0.30,
              global_damage_conf: float = 0.30,
              frame_skip: int    = 1,
              preview: bool      = False,
              save_parts: bool   = True,
              save_damage: bool  = True):
    """
    Process a video file through the two-stage car damage detection pipeline.

    Parameters
    ----------
    video_path        : input video file
    output_path       : base path for output videos (suffixes _parts / _damage added)
    global_parts_conf : minimum raw confidence passed to the parts YOLO model
    global_damage_conf: minimum raw confidence passed to the damage YOLO model
    frame_skip        : process every Nth frame (1 = every frame, 2 = every other, …)
    preview           : show live preview windows while processing
    save_parts        : write the parts-annotated output video
    save_damage       : write the damage-annotated output video
    """
    print(f"\n{'='*60}")
    print(f"  Car Damage Detection — Video Pipeline")
    print(f"  Input  : {video_path}")
    print(f"  Output : {output_path}  (frame_skip={frame_skip})")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps      = src_fps / frame_skip   # preserve original playback speed

    print(f"  Resolution : {width}×{height}")
    print(f"  Source FPS : {src_fps:.2f}   Output FPS: {out_fps:.2f}")
    print(f"  Frames     : {total_frames}  (processing every {frame_skip})\n")

    # Set up writers
    base, ext = os.path.splitext(output_path)
    # Force mp4 extension for broad compatibility
    out_ext = '.mp4'
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')

    parts_path  = f"{base}_parts{out_ext}"
    damage_path = f"{base}_damage{out_ext}"

    writer_parts  = None
    writer_damage = None

    if save_parts:
        writer_parts = cv2.VideoWriter(parts_path, fourcc, out_fps, (width, height))
    if save_damage:
        writer_damage = cv2.VideoWriter(damage_path, fourcc, out_fps, (width, height))

    frame_idx     = 0
    written_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Skip frames if requested
            if (frame_idx - 1) % frame_skip != 0:
                continue

            print(f"  Processing frame {frame_idx}/{total_frames} …", end='\r')

            parts_frame, damage_frame = process_frame(
                frame, global_parts_conf, global_damage_conf
            )

            # Burn HUD into both outputs
            draw_hud(parts_frame,  frame_idx, out_fps, total_frames, 'PARTS')
            draw_hud(damage_frame, frame_idx, out_fps, total_frames, 'DAMAGE')

            if writer_parts  is not None:
                writer_parts.write(parts_frame)
            if writer_damage is not None:
                writer_damage.write(damage_frame)

            written_count += 1

            # Optional live preview
            if preview:
                cv2.imshow('Parts Detection',  parts_frame)
                cv2.imshow('Damage Detection', damage_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  [Preview] 'q' pressed — stopping early.")
                    break

    finally:
        cap.release()
        if writer_parts  is not None:
            writer_parts.release()
        if writer_damage is not None:
            writer_damage.release()
        if preview:
            cv2.destroyAllWindows()

    print(f"\n\n  Processed {written_count} frames.")
    if save_parts:
        print(f"  Parts video  → {parts_path}")
    if save_damage:
        print(f"  Damage video → {damage_path}")
    print("  Done.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Two-stage car damage detection — VIDEO version',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Per-class thresholds are set inside the script in DAMAGE_THRESHOLDS and
PART_THRESHOLDS — edit those values to tune each class independently.
+*9875475444444444\


Examples:
  # Basic usage — process every frame
  python video_damage_detection.py car_video.mp4

  # Process every 3rd frame (faster, lower quality)
  python video_damage_detection.py car_video.mp4 --frame-skip 3

  # Custom output path + live preview
  python video_damage_detection.py car_video.mp4 --output out.mp4 --preview
-skip
  # Lower thresholds for more detections
  python video_damage_detection.py car_video.mp4 --parts-conf 0.10 --damage-conf 0.10

  # Only save the damage video
  python video_damage_detection.py car_video.mp4 --no-parts
        """
    )
    ap.add_argument('video',
                    help='Path to input video (e.g. car_clip.mp4)')
    ap.add_argument('--output', default='result_video.mp4',
                    help='Base output path — _parts / _damage suffixes are added  (default: result_video.mp4)')
    ap.add_argument('--parts-conf', type=float, default=0.30,
                    help='Global floor conf for part detection  (default: 0.30)')
    ap.add_argument('--damage-conf', type=float, default=0.30,
                    help='Global floor conf for damage detection  (default: 0.30)')
    ap.add_argument('--frame-skip', type=int, default=1,
                    help='Process every Nth frame to speed up inference  (default: 1 = every frame)')
    ap.add_argument('--preview', action='store_true',
                    help='Show live preview windows while processing (press q to stop)')
    ap.add_argument('--no-parts', action='store_true',
                    help='Skip writing the parts-annotated output video')
    ap.add_argument('--no-damage', action='store_true',
                    help='Skip writing the damage-annotated output video')
    args = ap.parse_args()

    run_video(
        video_path         = args.video,
        output_path        = args.output,
        global_parts_conf  = args.parts_conf,
        global_damage_conf = args.damage_conf,
        frame_skip         = args.frame_skip,
        preview            = args.preview,
        save_parts         = not args.no_parts,
        save_damage        = not args.no_damage,
    )