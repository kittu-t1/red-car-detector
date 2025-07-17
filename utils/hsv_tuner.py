import cv2
import numpy as np

def nothing(x):
    pass

# Use 0 for webcam or replace with a filename to tune on an image
cap = cv2.VideoCapture(r'.\assets\traffic_fixed.mp4')

cv2.namedWindow('HSV Tuner')
for param in ['H min','S min','V min','H max','S max','V max']:
    cv2.createTrackbar(param, 'HSV Tuner',
                       0 if 'min' in param else 255,
                       255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hmin = cv2.getTrackbarPos('H min', 'HSV Tuner')
    smin = cv2.getTrackbarPos('S min', 'HSV Tuner')
    vmin = cv2.getTrackbarPos('V min', 'HSV Tuner')
    hmax = cv2.getTrackbarPos('H max', 'HSV Tuner')
    smax = cv2.getTrackbarPos('S max', 'HSV Tuner')
    vmax = cv2.getTrackbarPos('V max', 'HSV Tuner')

    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, mask_bgr, result))
    cv2.imshow('HSV Tuner', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# After loop (whether you hit ESC or the video ends), print sliders:
print(f"Final range: lower={lower.tolist()}, upper={upper.tolist()}")



