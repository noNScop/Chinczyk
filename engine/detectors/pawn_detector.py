import cv2
import numpy as np

class PawnDetector:
    @staticmethod
    def find_green_pawns(frame_bgr):
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower_green = np.array([80, 100, 0])
        upper_green = np.array([100, 255, 80])

        green_mask = cv2.inRange(frame_hsv, lower_green, upper_green)

        return PawnDetector._get_coords(green_mask)
    
    @staticmethod
    def find_blue_pawns(frame_bgr):
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([108, 200, 60])
        upper_blue = np.array([120, 255, 130])

        blue_mask = cv2.inRange(frame_hsv, lower_blue, upper_blue)

        return PawnDetector._get_coords(blue_mask)
    
    @staticmethod
    def _get_coords(mask, min_area=500):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        pawn_centers = []

        for i in range(1, num_labels):  # skip background
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            if area < min_area:
                continue

            pawn_centers.append((cx, cy))

        return pawn_centers
