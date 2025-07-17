# main.py

import argparse
import os
import sys
import time
from typing import Union, List, Tuple, Dict

import cv2
import numpy as np
from ultralytics import YOLO

from utils.color_utils import get_color_ratio, COLOR_NAMES
from utils.llm_utils import extract_colors_from_text  # <<– New import
# (The rest of your imports remain the same)


# COCO class IDs for vehicles (as per Ultralytics YOLO “coco128” set)
VEHICLE_CLASSES = {
    'car': 2,
    'truck': 7,
    'bus': 5,
    'motorcycle': 3
}


class EnhancedCarTracker:
    """
    Centroid‐based tracker with:
      - Configurable dist_thresh & max_missed
      - Vehicle‐type awareness
      - Counting unique IDs + color‐matched IDs
    """
    def __init__(self, dist_thresh: float = 150, max_missed: int = 10):
        self.dist_thresh = dist_thresh
        self.max_missed = max_missed
        self.next_id = 1
        # tracked: id -> {'centroid':(x,y), 'missed':int, 'vehicle_type':str, 'first_seen', 'last_seen'}
        self.tracked: Dict[int, Dict] = {}
        self.all_ids = set()
        self.color_ids = set()
        self.frame_count = 0
        self.total_detections = 0
        self.color_matches = 0

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], bool, str]]) -> Dict[int, Dict]:
        """
        detections: [ ((x1,y1,x2,y2), color_matched, vehicle_type_str), … ]
        Returns dict of { id: data_dict } for currently‐tracked objects.
        """
        self.frame_count += 1

        # 1) Mark everyone as “missed” by 1
        for obj_id in list(self.tracked.keys()):
            self.tracked[obj_id]['missed'] += 1

        current_objs: Dict[int, Dict] = {}

        # 2) For each new detection, compute centroid & match to existing ID if close
        for (bbox, color_matched, vehicle_type) in detections:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            best_id = None
            best_dist = float('inf')

            for obj_id, data in self.tracked.items():
                px, py = data['centroid']
                dist = np.hypot(cx - px, cy - py)
                # Reward same‐type match slightly
                type_bonus = 0.8 if data.get('vehicle_type') == vehicle_type else 1.0
                adj_dist = dist * type_bonus
                if adj_dist < self.dist_thresh and adj_dist < best_dist:
                    best_dist = adj_dist
                    best_id = obj_id

            if best_id is not None:
                # Update existing
                self.tracked[best_id].update({
                    'centroid': (cx, cy),
                    'missed': 0,
                    'vehicle_type': vehicle_type,
                    'last_seen': self.frame_count
                })
                if color_matched:
                    self.color_ids.add(best_id)
                current_objs[best_id] = self.tracked[best_id]
                print(f"[DEBUG] → Frame {self.frame_count}: bbox={bbox}, color_matched={color_matched} → assigned existing ID={best_id} (dist={best_dist:.1f})")
            else:
                # Create new ID
                new_id = self.next_id
                self.next_id += 1
                self.tracked[new_id] = {
                    'centroid': (cx, cy),
                    'missed': 0,
                    'vehicle_type': vehicle_type,
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count
                }
                self.all_ids.add(new_id)
                if color_matched:
                    self.color_ids.add(new_id)
                current_objs[new_id] = self.tracked[new_id]
                print(f"[DEBUG] → Frame {self.frame_count}: bbox={bbox}, color_matched={color_matched} → assigned NEW ID={new_id}")

        # 3) Remove any ID with too many “missed” frames
        for obj_id in list(self.tracked.keys()):
            if self.tracked[obj_id]['missed'] > self.max_missed:
                print(f"[DEBUG] → Removing ID={obj_id} after missed={self.tracked[obj_id]['missed']}")
                del self.tracked[obj_id]

        # 4) Update stats
        self.total_detections += len(detections)
        self.color_matches += sum(1 for (_, cm, _) in detections if cm)

        return current_objs

    def get_statistics(self) -> Dict[str, int]:
        return {
            'total_unique_vehicles': len(self.all_ids),
            'color_matched_vehicles': len(self.color_ids),
            'currently_tracked': len(self.tracked),
            'total_detections': self.total_detections,
            'color_matches': self.color_matches,
            'frames_processed': self.frame_count
        }


def draw_enhanced_annotations(
    frame: np.ndarray,
    current_objs: Dict[int, Dict],
    detections: List[Tuple[Tuple[int, int, int, int], bool, str]],
    target_color: str
) -> np.ndarray:
    """
    Draw bounding boxes, ID labels, and color indicators.
    Red box if color_matched==True, Blue otherwise.
    White text on solid‐color background. ●=active, ○=recently‐missed.
    """
    # Build map from obj_id to its detection tuple
    detection_map: Dict[int, Tuple[Tuple[int, int, int, int], bool, str]] = {}
    for det in detections:
        bbox, color_matched, vehicle_type = det
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        for obj_id, data in current_objs.items():
            px, py = data['centroid']
            if abs(cx - px) < 20 and abs(cy - py) < 20:
                detection_map[obj_id] = det
                break

    for obj_id, data in current_objs.items():
        if obj_id not in detection_map:
            continue
        bbox, color_matched, vehicle_type = detection_map[obj_id]
        x1, y1, x2, y2 = bbox
        box_color = (0, 0, 255) if color_matched else (255, 0, 0)
        thickness = 3 if data['missed'] == 0 else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Label: ● or ○, ID, vehicle_type, [COLOR] if matched
        indicator = "●" if data['missed'] == 0 else "○"
        label = f"{indicator} ID:{obj_id} {vehicle_type.upper()}"
        if color_matched:
            label += f" [{target_color.upper()}]"

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def process_frame_enhanced(
    frame: np.ndarray,
    model: YOLO,
    tracker: EnhancedCarTracker,
    target_color: str,
    conf_threshold: float = 0.3,
    color_ratio_thresh: float = 0.10,
    vehicle_types: List[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Detect vehicles with YOLO, crop each box, compute get_color_ratio on that crop,
    update tracker, draw annotations, add stats overlay.
    Returns (annotated_frame, stats_dict).
    """
    if vehicle_types is None:
        vehicle_types = [VEHICLE_CLASSES['car']]
    detections: List[Tuple[Tuple[int, int, int, int], bool, str]] = []

    try:
        # 1) Run YOLO
        results = model(frame, verbose=False, conf=conf_threshold)
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id not in vehicle_types or conf < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                h, w = frame.shape[:2]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w-1), min(y2, h-1)
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                # 2) Compute color ratio on the crop
                crop = frame[y1:y2, x1:x2]
                try:
                    ratio = get_color_ratio(crop, target_color)
                    color_matched = ratio > color_ratio_thresh
                except Exception as e:
                    print(f"[WARNING] Color analysis error on crop {x1,y1,x2,y2}: {e}")
                    color_matched = False

                print(f"[DEBUG] Frame {tracker.frame_count}: Detected {target_color} ratio={ratio:.3f} in bbox=({x1},{y1},{x2},{y2}) → {'MATCHED' if color_matched else 'NO'}")

                vehicle_type_str = next(
                    (name for name, vid in VEHICLE_CLASSES.items() if vid == cls_id),
                    "unknown"
                )
                detections.append(((x1, y1, x2, y2), color_matched, vehicle_type_str))

        # 3) Update tracker
        current_objs = tracker.update(detections)

        # 4) Draw annotations
        annotated = draw_enhanced_annotations(frame.copy(), current_objs, detections, target_color)

        # 5) Overlay stats (include “this‐frame black count”)
        stats = tracker.get_statistics()
        current_color_count = sum(1 for (_, matched, _) in detections if matched)
        stats_lines = [
            f"Total Unique Vehicles: {stats['total_unique_vehicles']}",
            f"{target_color.title()} (this frame): {current_color_count}",
            f"Tracking Now: {stats['currently_tracked']}",
            f"Frame #: {stats['frames_processed']}"
        ]
        y_offset = 30
        for i, line in enumerate(stats_lines):
            y = y_offset + 25*i
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (5, y-th-5), (5+tw+5, y+5), (0,0,0), -1)
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        return annotated, stats

    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        return frame, {}


def setup_video_writer(output_path: str, cap: cv2.VideoCapture) -> Union[cv2.VideoWriter, None]:
    """
    Try multiple codecs; return a working VideoWriter or None if all fail.
    """
    if not output_path:
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for codec in ['mp4v', 'XVID', 'MJPG', 'H264']:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"[INFO] Writer initialized with codec: {codec}")
                return writer
            writer.release()
        except Exception:
            continue

    print("[ERROR] Could not initialize video writer with any codec.")
    return None


def main_enhanced(
    source: Union[str, int] = 0,
    output_path: str = None,
    target_color: str = "red",
    conf_threshold: float = 0.3,
    color_threshold: float = 0.10,
    skip_frames: int = 0,
    vehicle_types: List[str] = None,
    max_frames: int = None
) -> bool:
    """
    Main entrypoint: open video, run detection/tracking, display & optionally save annotated output.
    """
    # 1) Validate source
    if isinstance(source, str) and source != "0" and not os.path.exists(source):
        print(f"[ERROR] Source '{source}' not found.")
        return False
    if target_color not in COLOR_NAMES:
        print(f"[ERROR] Invalid color '{target_color}'. Choose from {COLOR_NAMES}.")
        return False

    # 2) Open video capture
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[ERROR] Could not open video source.")
            return False
    except Exception as e:
        print(f"[ERROR] Video capture init failed: {e}")
        return False

    # 3) Load YOLO model & create tracker
    try:
        print("[INFO] Loading YOLO model...")
        model = YOLO("yolo11n.pt")
        tracker = EnhancedCarTracker(dist_thresh=150, max_missed=10)
        print("[INFO] Model and tracker initialized.")
    except Exception as e:
        print(f"[ERROR] Model init failed: {e}")
        cap.release()
        return False

    # 4) Setup writer if requested
    writer = setup_video_writer(output_path, cap)

    # 5) Convert vehicle_types from strings to COCO IDs
    if vehicle_types is None:
        vehicle_types = ["car"]
    vehicle_type_ids: List[int] = []
    for vt in vehicle_types:
        if vt in VEHICLE_CLASSES:
            vehicle_type_ids.append(VEHICLE_CLASSES[vt])
        else:
            print(f"[WARNING] Unknown vehicle type: {vt}")
    if not vehicle_type_ids:
        print("[ERROR] No valid vehicle types specified.")
        cap.release()
        return False

    print(f"[INFO] Detecting color='{target_color}' on types={vehicle_types}")

    frame_count = 0
    processed_frames = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break

            frame_count += 1
            if frame_count % (skip_frames + 1) != 0:
                continue

            if max_frames and processed_frames >= max_frames:
                print(f"[INFO] Reached max_frames={max_frames}.")
                break

            annotated_frame, stats = process_frame_enhanced(
                frame, model, tracker,
                target_color=target_color,
                conf_threshold=conf_threshold,
                color_ratio_thresh=color_threshold,
                vehicle_types=vehicle_type_ids
            )
            processed_frames += 1

            if processed_frames % 30 == 0:
                elapsed = time.time() - start_time
                fps = processed_frames / elapsed if elapsed > 0 else 0.0
                print(f"[INFO] Processed {processed_frames} frames @ {fps:.1f} FPS.")

            cv2.imshow("Enhanced Vehicle Color Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User requested quit.")
                break

            if writer:
                writer.write(annotated_frame)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # 6) Print final summary
    elapsed_time = time.time() - start_time
    stats = tracker.get_statistics()
    print("\n" + "="*50)
    print("FINAL DETECTION RESULTS")
    print("="*50)
    print(f"Total frames read: {frame_count}")
    print(f"Frames processed: {processed_frames}")
    print(f"Processing time (s): {elapsed_time:.2f}")
    if elapsed_time > 0:
        print(f"Average FPS: {processed_frames/elapsed_time:.2f}")
    print(f"Total unique vehicles: {stats['total_unique_vehicles']}")
    print(f"{target_color.title()} vehicles (unique IDs): {stats['color_matched_vehicles']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Color matches (all detections): {stats['color_matches']}")
    if stats['total_unique_vehicles'] > 0:
        pct = 100.0 * stats['color_matched_vehicles'] / stats['total_unique_vehicles']
        print(f"Percentage '{target_color}': {pct:.1f}%")
    if output_path:
        print(f"Output saved to: {output_path}")
    print("="*50 + "\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced Vehicle Color Detection (LLM‐driven)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source traffic_fixed.mp4 --query "Show me black cars" --output ./outputs/black_cars.avi
  python main.py --source 0 --query "Detect red cars only" --color-thresh 0.1
  python main.py --source traffic_fixed.mp4 --skip-frames 2 --max-frames 200
        """
    )

    parser.add_argument(
        "--source", default="0",
        help="Video file path or camera index (default: 0)"
    )
    parser.add_argument(
        "--query", required=True,
        help="Natural-language query (e.g. \"Show me red cars and silver vehicles\")"
    )
    parser.add_argument(
        "--output", help="Output video file path (e.g. ./outputs/black_cars.avi)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="YOLO detection confidence threshold (0.0–1.0)"
    )
    parser.add_argument(
        "--color-thresh", type=float, default=0.10,
        help="Ratio threshold for deciding if crop is that color (0.0–1.0)"
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Skip this many frames between each processed frame"
    )
    parser.add_argument(
        "--max-frames", type=int,
        help="Maximum number of frames to process (for testing)"
    )
    parser.add_argument(
        "--vehicle-types", nargs="+", default=["car"],
        choices=list(VEHICLE_CLASSES.keys()),
        help="Which vehicle types to detect (choices: car truck bus motorcycle)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug/verbose logging"
    )

    args = parser.parse_args()
    if args.verbose:
        os.environ["ULTRALYTICS_LOG_LEVEL"] = "DEBUG"

    source = int(args.source) if args.source.isdigit() else args.source

    # —————————— NEW: LLM color‐extraction from args.query ——————————
    extracted = extract_colors_from_text(args.query)
    if not extracted:
        print("[ERROR] The LLM did not find any valid color in your query.")
        print("Allowed colors are:", COLOR_NAMES)
        sys.exit(1)

    if len(extracted) > 1:
        print(f"[WARNING] LLM returned multiple colors {extracted}. Using the first one: '{extracted[0]}'")
    target_color = extracted[0]
    print(f"[INFO] Tracking color = '{target_color}'")

    # —————— Call the existing enhanced pipeline with that single color ——————
    success = main_enhanced(
        source=source,
        output_path=args.output,
        target_color=target_color,
        conf_threshold=args.conf,
        color_threshold=args.color_thresh,
        skip_frames=args.skip_frames,
        vehicle_types=args.vehicle_types,
        max_frames=args.max_frames
    )
    sys.exit(0 if success else 1)
