import sys
import cv2
import os

def remux_mp4(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ Input not found: {input_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {input_path}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"🔄 Remuxing {input_path} → {output_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    print("✅ Remux complete.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_mp4.py <in.mp4> <out_fixed.mp4>")
        sys.exit(1)
    remux_mp4(sys.argv[1], sys.argv[2])
