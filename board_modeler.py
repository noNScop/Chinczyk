#create similar to the above editor where /home/stick/sesm5/CV/Martyn/project2/Chinczyk/data/board.jpg image will be shown and i will be able to click it. all clicks will be registered the aim is to mark all tiles on ludo board with their position and id so first when i click there will be registered click for tile with id 0; when i click enter button then the id registered will be increased by 1 current id will be displayed somewhere on image; all points already clicked will be shown; all scored will be saved in dict in form id: [[x1,y1], [x2,y2], ...] and when escape wil be pressed then program will terminate and dict will be saved so it can be loaded later. point with cooridnates 0,0 is point in the center of the image
'''enter for next tile ID
click on some place to annotate the place
escape to save everything to file'''
import cv2
import numpy as np
import os
from engine.helpers import find_main_folder

main_folder = find_main_folder()

def next_free_filename(base_path):
    i = 0
    while True:
        path = base_path.replace(".npy", f"_{i}.npy")
        if not os.path.exists(path):
            return path
        i += 1

# -------- CONFIG --------
IMAGE_PATH = os.path.join(main_folder, "data", "board.jpg") 

os.makedirs(os.path.join(main_folder, "engine", "board_data"), exist_ok=True)
SAVE_PATH = next_free_filename(os.path.join(main_folder, "engine", "board_data", "board_tiles.npy"))

WINDOW_NAME = "Board annotator"
BOARD_SIZE = 800  # coordinate system 0..800
# ------------------------

# Load image
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)

h, w = img_bgr.shape[:2]

# Scaling factors from image pixels → board coordinates
scale_x = BOARD_SIZE / w
scale_y = BOARD_SIZE / h

# State
current_id = 0
points_dict = {}  # id -> list of [x, y] in board coordinates
display = img_bgr.copy()

# ---------------- GUI & DRAW ----------------
def draw_overlay():
    global display
    display = img_bgr.copy()

    # Draw all points
    for tile_id, pts in points_dict.items():
        for x, y in pts:
            px = int(x / scale_x)
            py = int(y / scale_y)
            cv2.circle(display, (px, py), 4, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(tile_id),
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

    # Draw current id info
    cv2.putText(
        display,
        f"Current tile ID: {current_id}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2
    )

def on_mouse(event, x, y, flags, param):
    global points_dict

    if event == cv2.EVENT_LBUTTONDOWN:
        # convert to board coordinates (0,0 top-left, 800,800 bottom-right)
        xb = int(x * scale_x)
        yb = int(y * scale_y)

        points_dict.setdefault(current_id, []).append([xb, yb])
        print(f"Tile {current_id}: added point ({xb}, {yb})")
        draw_overlay()

# ---------------- WINDOW ----------------
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1000, 1000)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

draw_overlay()

while True:
    cv2.imshow(WINDOW_NAME, display)
    key = cv2.waitKey(1) & 0xFF

    # ENTER → next tile ID
    if key == 13:
        current_id += 1
        print(f"Switched to tile ID {current_id}")

    # ESC → save and exit
    elif key == 27:
        np.save(SAVE_PATH, points_dict)
        print(f"Saved tile data to {SAVE_PATH}")
        break

cv2.destroyAllWindows()
