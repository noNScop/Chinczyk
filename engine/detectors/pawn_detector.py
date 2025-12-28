import cv2
import numpy as np

class PawnDetector:
    """
    Detects pawn positions based on color segmentation in HSV space.
    Designed for board-game pawns with relatively consistent colors and sizes.
    """

    @staticmethod
    def _green_mask(frame_bgr, low, high):
        """
        Create a binary mask for green-colored regions.

        Parameters:
        - frame_bgr: input frame in BGR color space
        - low, high: HSV lower and upper bounds

        Returns:
        - Binary mask highlighting green regions
        """

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array(low), np.array(high))

    @staticmethod
    def _blue_mask(frame_bgr, low, high):
        """
        Create a binary mask for blue-colored regions.

        Parameters:
        - frame_bgr: input frame in BGR color space
        - low, high: HSV lower and upper bounds

        Returns:
        - Binary mask highlighting blue regions
        """

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array(low), np.array(high))
    
    @staticmethod
    def find_green_pawns(frame_bgr):
        """
        Detect green pawns in the frame.

        Uses a coarse HSV mask first, followed by connected component analysis.
        """

        mask = PawnDetector._green_mask(
            frame_bgr,
            low=[80, 100, 0],
            high=[100, 255, 160],
        )
        return PawnDetector._get_coords(frame_bgr, mask, "green")

    @staticmethod
    def find_blue_pawns(frame_bgr):
        """
        Detect blue pawns in the frame.

        Uses a coarse HSV mask first, followed by connected component analysis.
        """

        mask = PawnDetector._blue_mask(
            frame_bgr,
            low=[108, 100, 60],
            high=[120, 255, 160],
        )
        return PawnDetector._get_coords(frame_bgr, mask, "blue")
    
    @staticmethod
    def _get_coords(frame_bgr, mask, color, min_area=500):
        """
        Extract pawn center coordinates from a binary mask.

        Workflow:
        1. Find connected components in the mask
        2. Filter out small components (noise)
        3. For large blobs, re-segment with stricter HSV thresholds
           to split merged pawns
        4. Compute centroids of final components

        Parameters:
        - frame_bgr: original frame
        - mask: binary mask for pawn color
        - color: 'green' or 'blue' (controls refinement thresholds)
        - min_area: minimum area to consider a component a pawn

        Returns:
        - List of (x, y) pawn center coordinates
        """

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        pawn_centers = []

        # Skip label 0 (background)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            # Ignore small components (noise)
            if area < min_area:
                continue

            # Large components may represent multiple overlapping pawns
            if area > 3000:
                roi = frame_bgr[y:y+h, x:x+w]

                # Apply stricter HSV thresholds locally to split merged blobs
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

                # Extract centroids of refined components
                for j in range(1, n2):
                    a2 = stats2[j, cv2.CC_STAT_AREA]
                    if a2 < min_area:
                        continue

                    cx2, cy2 = centroids2[j]
                    pawn_centers.append((x + cx2, y + cy2))

            else:
                # Single pawn detected
                pawn_centers.append((cx, cy))

        return pawn_centers