import cv2
import numpy as np


import os

FILE = "some_frame.png"

def load_img(path, frame_idx=0):
    """
    Load image from:
    - image file (.png, .jpg, ...)
    - video file (.mp4, .avi, ...) → returns frame_idx frame

    Returns: BGR image (np.ndarray)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    # ---- IMAGE ----
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        return img

    # ---- VIDEO ----
    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx} from {path}")

        return frame

    raise ValueError(f"Unsupported file type: {ext}")






folder = os.path.dirname(os.path.abspath(__file__))
file_fol = os.path.join(folder, "data", FILE)
img_bgr = load_img(file_fol)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)




def nothing(x):
    pass

cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mask", 800, 450)

cv2.createTrackbar("H min", "mask", 0, 255, nothing)
cv2.createTrackbar("H max", "mask", 179, 255, nothing) # dziwne że tu da się większą wartość niż 179 dać
cv2.createTrackbar("S min", "mask", 0, 255, nothing)
cv2.createTrackbar("S max", "mask", 255, 255, nothing)
cv2.createTrackbar("V min", "mask", 0, 255, nothing)
cv2.createTrackbar("V max", "mask", 255, 255, nothing)

while True:
    h_min = cv2.getTrackbarPos("H min", "mask")
    h_max = cv2.getTrackbarPos("H max", "mask")
    s_min = cv2.getTrackbarPos("S min", "mask")
    s_max = cv2.getTrackbarPos("S max", "mask")
    v_min = cv2.getTrackbarPos("V min", "mask")
    v_max = cv2.getTrackbarPos("V max", "mask")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower, upper)

    print(lower)
    print(upper)
    cv2.imshow("mask", mask)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()