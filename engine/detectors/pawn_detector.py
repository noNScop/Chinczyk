import cv2
import numpy as np

class PawnDetector:

    @staticmethod
    def _green_mask(frame_bgr, low, high):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array(low), np.array(high))

    @staticmethod
    def _blue_mask(frame_bgr, low, high):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array(low), np.array(high))
    
    @staticmethod
    def find_green_pawns(frame_bgr):
        mask = PawnDetector._green_mask(
            frame_bgr,
            low=[80, 100, 0],
            high=[100, 255, 160],
        )
        return PawnDetector._get_coords(frame_bgr, mask, "green")

    @staticmethod
    def find_blue_pawns(frame_bgr):
        mask = PawnDetector._blue_mask(
            frame_bgr,
            low=[108, 100, 60],
            high=[120, 255, 160],
        )
        return PawnDetector._get_coords(frame_bgr, mask, "blue")
    
    @staticmethod
    def _get_coords(frame_bgr, mask, color, min_area=500):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        pawn_centers = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            if area < min_area:
                continue

            if area > 3000:
                roi = frame_bgr[y:y+h, x:x+w]

                if color == "green":
                    precise_mask = PawnDetector._green_mask(
                        roi,
                        low=[80, 100, 0],
                        high=[100, 255, 80],
                    )
                else:
                    precise_mask = PawnDetector._blue_mask(
                        roi,
                        low=[108, 100, 60],
                        high=[120, 255, 120],
                    )

                n2, _, stats2, centroids2 = cv2.connectedComponentsWithStats(precise_mask)

                for j in range(1, n2):
                    a2 = stats2[j, cv2.CC_STAT_AREA]
                    if a2 < min_area:
                        continue

                    cx2, cy2 = centroids2[j]
                    pawn_centers.append((x + cx2, y + cy2))

            else:
                pawn_centers.append((cx, cy))

        return pawn_centers