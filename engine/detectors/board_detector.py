import cv2
import numpy as np
from itertools import combinations
import math

class BoardDetector:
    def __init__(self, canonical_size = 800):
        self.canonical_size = canonical_size

        self.M = None
        self.board_corners = None
        #self.countur = None
        self.M = None
        self.M_inv = None
        self.vertices = None


    def update(self, frame):
        """
        Detect board and compute warp matrices.
        """
        self.frame_shape = frame.shape[:2]

        self.board = self.find_board_contour(frame)
        
        self.detected_lines =  self.detect_edges_new(self.board)
        self.vertices = self.choose_edges(self.board)

        if self.vertices is None or len(self.vertices) != 4:
            return
        
        if self.board is None:
            return
        
        contour = np.array(self.vertices, dtype=np.float32).reshape(-1, 1, 2)
        board_corners = self.get_corners_ordered(contour) # f boarder not fully visible i dont update its parameters
        if board_corners is None:
            return
        self.board_corners = board_corners

        dst = np.array([
            [0, 0],
            [self.canonical_size, 0],
            [self.canonical_size, self.canonical_size],
            [0, self.canonical_size]
        ], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(self.board_corners, dst)
        self.M_inv = np.linalg.inv(self.M)


    def find_board_contour(self, frame_bgr): # returns cords of board corners
        frame_bgr = frame_bgr.copy()
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([13, 60, 110])
        upper_bound = np.array([100, 255, 255])
        board = cv2.inRange(frame_hsv, lower_bound, upper_bound)

        kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # close gaps inside board
        board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel_close)

        # remove small noise
        board = cv2.morphologyEx(board, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(
            board,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        
        board_contour = max(contours, key=cv2.contourArea)
        self.lines = None
        
        hull = cv2.convexHull(board_contour)

        mask = np.zeros(board.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # self.board = mask     

        return mask

    
    @staticmethod
    def get_corners_ordered(contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

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
    

    def detect_edges_new(self, mask):
        """
        Detect edges and lines by fitting a polygon to the board mask.
        Stores lines in self.detected_lines.
        """
        # 1. Find contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.detected_lines = None
            return None

        # 2. Take the largest contour
        board_contour = max(contours, key=cv2.contourArea)

        # 3. Approximate polygon
        epsilon = 0.005 * cv2.arcLength(board_contour, True)  # adjust for smoothness
        polygon = cv2.approxPolyDP(board_contour, epsilon, True)

        self.detected_polygon = polygon  # store polygon for visualization

        # 4. Extract lines from polygon edges
        lines = []
        pts = polygon.reshape(-1, 2)
        for i in range(len(pts)):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[(i + 1) % len(pts)])  # wrap around
            lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])

        detected_lines = np.array(lines).reshape(-1, 1, 4)  # same format as HoughLinesP
        return detected_lines


    def line_intersection(self, l1, l2, mask=None):
        """Find intersection point of two lines (x1,y1,x2,y2 format).
        Returns None if intersection is too far from mask bounding box."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1*x1 + B1*y1

        A2 = y4 - y3
        B2 = x3 - x4
        C2 = A2*x3 + B2*y3

        det = A1*B2 - A2*B1
        if abs(det) < 1e-6:
            return None  # lines are parallel

        x = (B2*C1 - B1*C2) / det
        y = (A1*C2 - A2*C1) / det

        # optional: check if inside mask bounding box
        if mask is not None:
            h, w = mask.shape
            margin = max(h, w) * 0.2  # allow intersection slightly outside
            #print("x,y", x, y, margin)
            if not (-margin <= x <= w + margin and -margin <= y <= h + margin):
                #print("out of bounds")
                return None

        #print("intersection", x, y)
        return (x, y)


    def angle_between(self, v1, v2):
        """Angle in degrees between two vectors."""
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1, 1)
        return np.degrees(np.arccos(cos_theta))

    def choose_edges(self, mask):
        '''idea is to extend line segments into lines and check intersections of all 4 lines combinations
        then check which quadrilateral is most like a square (sides similar length and angles close to 90 degrees)
        and return most square similar combinatoin of edges- places of their intersections are board corners'''
        if self.detected_lines is None or len(self.detected_lines) < 4:
            self.chosen_edges = None
            return

        lines = [l[0] for l in self.detected_lines]  # shape (N,4)
        best_score = float('inf')
        best_quad = None

        for quad in combinations(lines, 4):
            # compute intersections of all line pairs
            pts = []
            for i in range(4):
                for j in range(i+1, 4):
                    pt = self.line_intersection(quad[i], quad[j], mask = mask)
                    if pt is not None:
                        pts.append(pt)

            #print("pts", len(pts ),"\n", pts )

            if len(pts) != 4:
                continue  # not forming a quadrilateral

            pts = np.array(pts)

            # order points clockwise (centroid method)
            center = pts.mean(axis=0)
            def angle_from_center(p):
                return math.atan2(p[1]-center[1], p[0]-center[0])
            pts = sorted(pts, key=angle_from_center)
            pts = np.array(pts)

            # compute side lengths
            sides = [np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)]
            side_mean = np.mean(sides)
            side_var = np.var(sides)

            # compute angles
            angles = []
            for i in range(4):
                v1 = pts[i] - pts[i-1]
                v2 = pts[(i+1)%4] - pts[i]
                angles.append(self.angle_between(v1, v2))
            angle_var = np.var(angles)

            # score = variance of sides + variance of angles
            score = side_var + angle_var
            if score < best_score:
                best_score = score
                best_quad = quad
            # if variance too big of the best quad then we can discard the result

        # There is a case when best_quad is None and it causes an error, so it needs to be handled
        if best_quad is None:
            self.chosen_edges = None
            return

        self.chosen_edges = best_quad
        vertices = []
        for i in range(4):
            pt = self.line_intersection(self.chosen_edges[i],
                                                self.chosen_edges[(i+1)%4],
                                                mask=self.board)
            if pt is not None:
                vertices.append(pt)
        return vertices





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
    
    def points_inside_board(self, points_unwarped):
        """
        points: iterable of (x, y)
        returns: list of booleans
        """

        points = self.warp_points(points_unwarped)

        print(points_unwarped, " un")
        print(points)

        if self.board is None:
            raise RuntimeError("Board not detected")

        
        #h, w = self.board.shape
        #print("hwhwhwhw", h, w)
        h, w, = self.canonical_size, self.canonical_size
        results = []

        for x, y in points:
            x = int(round(x))
            y = int(round(y))

            if 0 <= x < w and 0 <= y < h:
                #results.append(self.board[y, x] > 0)
                results.append(True)
            else:
                results.append(False)

        return results

    @property
    def ready(self):
        return self.M is not None

    def get_M(self):
        return self.M

    def get_M_inv(self):
        return self.M_inv

