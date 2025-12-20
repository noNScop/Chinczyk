import cv2
import numpy as np
from detectors.die_handler import Die_handler
import matplotlib.pyplot as plt####


class PlayerTurnDieDetector:

    @staticmethod
    def circle_similarity(pts):
        return len(pts) # number of verices of shape #1 - 1/len(pts)
    
    @staticmethod
    def label_object(img_hsv, pts): # info from plansza_detector will be useful here
        # if_on_plansza = any(pts) inside PlanszaDetector.bounds
        circle_similarity = PlayerTurnDieDetector.circle_similarity(pts)
        area = cv2.contourArea(pts)
        dots_num, _ = Die_handler.count_num_on_die(img_hsv, pts)

        if area > 10000 and dots_num < 1 and circle_similarity == 4:
            return 'marker'
        elif area > 2700 and area < 4500 and circle_similarity >=4 and circle_similarity <= 6:
            return 'die'
        elif circle_similarity > 5 and area > 4500:
            return 'reflection'
        else:
            return 'none'



    @staticmethod
    def find_objects(frame_bgr):
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 185]) #hsv
        upper = np.array([255, 80, 255]) #hsv

        lower_bgr = np.array([135, 0, 160]) #bgr
        upper_bgr = np.array([255, 255, 255])#bgr
        mask = cv2.inRange(frame_bgr, lower_bgr, upper_bgr)

        # plt.figure(figsize=(6,6))
        # plt.imshow(mask, cmap='gray') ######
        # plt.title("HSV Mask")
        # plt.axis('off')
        # plt.show()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], []

        #convex houl
        hulls = [cv2.convexHull(cnt) for cnt in contours]
        hulls = sorted(hulls, key=cv2.contourArea, reverse=True)[:3] # top3 area houls are potential player_turn_marker, die and reflection
        pts_arr = []
        labels = []
        for cnt in hulls:
            area = cv2.contourArea(cnt)
            if area < 1000:  # throw away area if eg die is not on frame
                continue

            # Approximate contour to polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            pts = approx.reshape(-1, 2)
            label = PlayerTurnDieDetector.label_object(frame_hsv, pts)
            pts_arr.append(pts)
            labels.append(label)

            
        return pts_arr, labels

            


