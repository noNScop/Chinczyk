import cv2
import numpy as np

class Die_handler():
    def __init__(self, memory_frames = 30):
        self.number = 0
        self.memory_frames = memory_frames 
        self.circles = None
    
    @staticmethod
    def count_num_on_die(img_hsv, pts):
        #Create mask for the die polygon
        mask_die = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_die, [pts], 255)

        # Extract die region from original image
        die_region = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_die)

        #  Convert to grayscale for circle detection
        die_gray = cv2.cvtColor(die_region, cv2.COLOR_HSV2BGR)
        die_gray = cv2.cvtColor(die_gray, cv2.COLOR_BGR2GRAY)

        die_gray = cv2.GaussianBlur(die_gray, (5,5), 0)
        
        _, die_thresh = cv2.threshold(die_gray, 80, 255, cv2.THRESH_BINARY_INV)

        #  Detect circles using Hough Transform
        circles = cv2.HoughCircles(
            die_thresh,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=11,         
            param1=70,          
            param2=11,          
            minRadius=4,        
            maxRadius=9
        )

        if circles is not None:
            num_pips = circles.shape[1]
            return num_pips, circles
        else:
            return 0, None
    
    @staticmethod
    def num_on_die_with_memory(num_pips): 
        return num_pips # placeholder
    
    @staticmethod
    def circles_on_die_with_memory(circles):
        return np.round(circles[0]).astype(int)

    
    def update(self, frame_bgr, pts):
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        num_pips, circles = self.count_num_on_die(frame_hsv, pts)
        #uwzględnione też zerowe wyniki np kiedy kość się toczy
        if num_pips > 0:
            self.number = self.num_on_die_with_memory(num_pips)
            self.circles = self.circles_on_die_with_memory(circles)

        
    def get_number(self):
        return self.number
    
    def get_circes(self):
        return self.circles