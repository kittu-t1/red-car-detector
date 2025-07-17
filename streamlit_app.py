# streamlit_app.py

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

from main import EnhancedCarTracker
from utils.color_utils import get_color_ratio, COLOR_NAMES

# Setup
st.set_page_config(page_title="YOLO Car Color Detector", layout="wide")
BASE_DIR = Path(__file__).parent
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    wp = BASE_DIR / weights_path
    if not wp.exists():
        st.error(f"⚠️ Weights not found: {wp}")
        st.stop()
    return YOLO(str(wp))

def run_detection(input_path, weights, conf_thresh, color, ratio_thresh):
    # init model & tracker
    model   = load_model(weights)
    tracker = EnhancedCarTracker(dist_thresh=150, max_missed=10)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # prepare output writer
    stem     = Path(input_path).stem
    out_path = OUT_DIR / f"{stem}_detected.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # placeholder for live preview
    frame_slot = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model.predict(frame, conf=conf_thresh, verbose=False)[0]

        # build detections list: (bbox, color_matched, vehicle_type)
        detections = []
        for *xyxy, conf, cls in results.boxes.data.tolist():
            if int(cls) != 2:  # only class=2 (car)
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            matched = False
            if crop.size and get_color_ratio(crop, color) >= ratio_thresh:
                matched = True
            detections.append(((x1, y1, x2, y2), matched, "car"))

        # update tracker and draw boxes
        current = tracker.update(detections)
        for vid, data in current.items():
            bbox = data['centroid']  # not used for drawing
        # reuse your draw function or simply draw matched boxes:
        for (x1, y1, x2, y2), matched, _ in detections:
            if matched:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # live preview
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(rgb, caption="Live detection", use_column_width=True)

        writer.write(frame)

    # cleanup
    cap.release()
    writer.release()

    # gather stats
    stats = tracker.get_statistics()
    total_unique = stats['total_unique_vehicles']
    color_unique = stats['color_matched_vehicles']

    return str(out_path), total_unique, color_unique

def main():
    st.title("🚗 YOLO v11 Car–Color Detector")

    st.sidebar.header("Settings")
    weights     = st.sidebar.text_input("Weights file", "yolov11n.pt")
    conf_thresh = st.sidebar.slider("Confidence", 0.0, 1.0, 0.2, 0.01)
    color       = st.sidebar.selectbox("Color", list(COLOR_NAMES))
    ratio_thresh= st.sidebar.slider("Color‐ratio %", 0.0, 1.0, 0.08, 0.01)
    upload      = st.sidebar.file_uploader("Upload video", type=["mp4","avi","mov"])

    if upload:
        tmp_in = BASE_DIR / "temp" / upload.name
        tmp_in.parent.mkdir(exist_ok=True)
        tmp_in.write_bytes(upload.getbuffer())

        if st.sidebar.button("▶️ Run detection"):
            with st.spinner("Processing…"):
                out_path, total, matched = run_detection(
                    str(tmp_in), weights, conf_thresh, color, ratio_thresh
                )

            st.success(f"✅ Done! Saved to `{out_path}`")
            st.write(f"Detected **{matched}** unique “{color}” cars out of **{total}** total unique cars.")
            st.video(out_path)

    st.markdown(
        """
        - Outputs are saved to the `outputs/` folder.  
        - **total unique** = distinct vehicles tracked across all frames.  
        - **color unique** = how many of those ever matched your color threshold.
        """
    )

if __name__ == "__main__":
    main()
