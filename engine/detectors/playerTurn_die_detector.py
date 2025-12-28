import cv2
import numpy as np
from detectors.die_handler import Die_handler
import matplotlib.pyplot as plt


class PlayerTurnDieDetector:
    """
    Detects and classifies board objects related to turn handling:
    - player turn marker
    - die
    - light reflections (false positives)

    Detection is based on color thresholding, contour geometry,
    area heuristics, and die-dot analysis.
    """

    @staticmethod
    def circle_similarity(pts):
        """
        Simple shape heuristic based on polygon vertex count.

        Fewer vertices → more rectangular-like
        More vertices → more circular/irregular

        Returns:
        - Number of vertices in the approximated contour
        """
        return len(pts)
    
    @staticmethod
    def label_object(img_hsv, pts):
        """
        Classify a detected contour into one of:
        - 'marker'
        - 'die'
        - 'reflection'
        - 'none'

        Classification is based on:
        - contour area
        - polygon complexity (vertex count)
        - number of dots detected on a die face
        """

        circle_similarity = PlayerTurnDieDetector.circle_similarity(pts)
        area = cv2.contourArea(pts)

        # Count pips on the die face (if any)
        dots_num, _ = Die_handler.count_num_on_die(img_hsv, pts)

        # Player turn marker: large, rectangular, no dots
        if area > 10000 and dots_num < 1 and circle_similarity == 4:
            return 'marker'
        
        # Die: medium-sized, roughly square/circular, limited vertex count
        elif area > 2500 and area < 4500 and circle_similarity >=4 and circle_similarity <= 6:
            return 'die'
        
        # Reflection: large and irregular, many vertices
        elif circle_similarity > 5 and area > 4500:
            return 'reflection'
        
        return 'none'


    @staticmethod
    def find_objects(frame_bgr):
        """
        Detect and classify turn-related objects in the frame.

        Pipeline:
        1. Threshold candidate regions using both BGR and HSV color spaces
        2. Clean the mask with morphological operations
        3. Extract contours and convex hulls
        4. Keep largest candidates (marker, die, reflection)
        5. Classify each object geometrically and semantically
        """

        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # HSV range for yellow-ish / light objects
        lower_hsv = np.array([7, 13, 154])
        upper_hsv = np.array([19, 75, 255])

        # BGR range used as a complementary mask
        lower_bgr = np.array([135, 0, 160])
        upper_bgr = np.array([255, 255, 255])

        mask_bgr = cv2.inRange(frame_bgr, lower_bgr, upper_bgr)
        mask_hsv = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)

        # Clean up noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_bgr = cv2.morphologyEx(mask_bgr, cv2.MORPH_OPEN, kernel)
        mask_bgr = cv2.morphologyEx(mask_bgr, cv2.MORPH_CLOSE, kernel)

        # Combine masks from both color spaces
        mask_combined = np.logical_or(mask_bgr, mask_hsv).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], []

        # Use convex hulls to stabilize shape analysis
        hulls = [cv2.convexHull(cnt) for cnt in contours]

        # Keep only the largest candidates:
        # marker, die, and possible reflection
        hulls = sorted(hulls, key=cv2.contourArea, reverse=True)[:3]

        pts_arr = []
        labels = []

        for cnt in hulls:
            area = cv2.contourArea(cnt)

            # Discard very small regions (noise or partial visibility)
            if area < 1000:
                continue

            # Approximate contour to polygon for geometric analysis
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            pts = approx.reshape(-1, 2)

            label = PlayerTurnDieDetector.label_object(frame_hsv, pts)

            pts_arr.append(pts)
            labels.append(label)

        return pts_arr, labels

            


