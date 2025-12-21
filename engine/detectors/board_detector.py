import cv2
import numpy as np

class BoardDetector:
    def __init__(self, canonical_size = 800):
        self.canonical_size = canonical_size

        self.M = None
        self.board_corners = None
        self.countur = None
        self.M = None
        self.M_inv = None


    def update(self, frame):
        """
        Detect board and compute warp matrices.
        """
        contour = self.find_board_contour(frame)
        if contour is None:
            return
        board_corners = self.get_corners_ordered(contour)
        if board_corners is None:
            return
        self.board_corners = board_corners
        self.countur = contour

        dst = np.array([
            [0, 0],
            [self.canonical_size, 0],
            [self.canonical_size, self.canonical_size],
            [0, self.canonical_size]
        ], dtype=np.float32)

        print("corners######", self.board_corners)
        self.M = cv2.getPerspectiveTransform(self.board_corners, dst)
        self.M_inv = np.linalg.inv(self.M)


    @staticmethod
    def find_board_contour(frame_bgr): # returns cords of board corners
        frame_rgb = frame_bgr.copy()
        frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        lower_bound = np.array([0, 0, 140])
        upper_bound = np.array([255, 255, 255])
        board = cv2.inRange(frame_hsv, lower_bound, upper_bound)

        contours, _ = cv2.findContours(
            board,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        
        board_contour = max(contours, key=cv2.contourArea)
        return board_contour

    
    @staticmethod
    def get_corners_ordered(countur):
        peri = cv2.arcLength(countur, True)
        approx = cv2.approxPolyDP(countur, 0.01 * peri, True)

        try:
            pts = approx.reshape(4, 2)
        except: 
            return None # for now simple solution if board not fully visible
            

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        ordered = np.zeros((4,2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]       # top-left
        ordered[2] = pts[np.argmax(s)]       # bottom-right
        ordered[1] = pts[np.argmin(diff)]    # top-right
        ordered[3] = pts[np.argmax(diff)]    # bottom-left
        return ordered
    

    def warp(self, frame):
        return cv2.warpPerspective(frame, self.M, (self.canonical_size, self.canonical_size))


    def unwarp(self, warped_frame):
        h, w = self.frame_shape
        return cv2.warpPerspective(warped_frame, self.M_inv, (w, h))
    

    def warp_points(self, points):
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)
        return warped_pts
    

    def unwarp_points(self, points):
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        orig_pts = cv2.perspectiveTransform(pts, self.M_inv).reshape(-1, 2)
        return orig_pts

