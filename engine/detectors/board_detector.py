import cv2
import math
import numpy as np
from itertools import combinations

class BoardDetector:
    """
    Detects a board in the input frame, estimates a homography, and provides
    utilities for warping/unwarping frames and points into a canonical
    board coordinate system.
    """

    def __init__(self, canonical_size = 800):
        """
        Args:
            canonical_size (int):
                Size (in pixels) of the square canonical board after warping.
                The board is mapped to a canonical_size × canonical_size image.
        """

        self.canonical_size = canonical_size
        
        # Perspective transform matrices
        self.M = None          # image → canonical
        self.M_inv = None      # canonical → image

        # Board geometry
        self.board_corners = None   # ordered 4 corner points in image space
        self.vertices = None        # detected board vertices (unordered)
        self.frame_shape = None     # original frame shape


    def update(self, frame):
        """
        Detect the board in the given frame and compute perspective transforms.

        This method:
        1. Segments the board region.
        2. Detects candidate edges.
        3. Selects the best quadrilateral approximating a square.
        4. Computes forward and inverse homographies.

        If the board is not fully visible or detection fails, internal
        parameters are left unchanged.
        """

        self.frame_shape = frame.shape[:2]

        # Segment the board region
        self.board = self.find_board_contour(frame)
        
        # Detect edges and candidate vertices
        self.detected_lines =  self.detect_edges_new(self.board)
        self.vertices = self.choose_edges(self.board)

        # Require exactly 4 vertices to proceed
        if self.vertices is None or len(self.vertices) != 4:
            return
        
        if self.board is None:
            return
        
        contour = np.array(self.vertices, dtype=np.float32).reshape(-1, 1, 2)

        # Order corners consistently (TL, TR, BR, BL)
        board_corners = self.get_corners_ordered(contour) # f boarder not fully visible i dont update its parameters
        if board_corners is None:
            return
        
        self.board_corners = board_corners

        # Canonical square destination
        dst = np.array([
            [0, 0],
            [self.canonical_size, 0],
            [self.canonical_size, self.canonical_size],
            [0, self.canonical_size]
        ], dtype=np.float32)

        # Compute homographies
        self.M = cv2.getPerspectiveTransform(self.board_corners, dst)
        self.M_inv = np.linalg.inv(self.M)


    def find_board_contour(self, frame_bgr):
        """
        Segment the board area using HSV color thresholding and morphology.

        Returns:
            mask (uint8):
                Binary mask of the detected board region.
        """

        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # HSV color range for the board surface
        lower_bound = np.array([13, 60, 110])
        upper_bound = np.array([100, 255, 255])

        board = cv2.inRange(frame_hsv, lower_bound, upper_bound)

        # Morphological cleanup:
        # - close fills internal gaps
        # - open removes small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel)
        board = cv2.morphologyEx(board, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            board,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None
        
        # Keep the largest connected component
        board_contour = max(contours, key=cv2.contourArea)
        
        # Convex hull stabilizes contour for partial occlusions
        hull = cv2.convexHull(board_contour)

        mask = np.zeros(board.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)    

        return mask

    
    @staticmethod
    def get_corners_ordered(contour):
        """
        Approximate contour to 4 points and return them in a consistent order.

        Ordering:
            0 → top-left
            1 → top-right
            2 → bottom-right
            3 → bottom-left

        Returns:
            np.ndarray (4,2) or None if approximation fails.
        """

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

        try:
            pts = approx.reshape(4, 2)
        except Exception:
            # Board not fully visible or not quadrilateral
            return None

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
        Detect board edges by polygon approximation of the board mask.

        The resulting edges are stored in the same format as HoughLinesP
        (x1, y1, x2, y2).
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
        """
        Compute intersection point of two infinite lines.

        Returns None if:
        - lines are parallel
        - intersection lies far outside the board region
        """

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

            if not (-margin <= x <= w + margin and -margin <= y <= h + margin):
                return None

        return (x, y)


    def angle_between(self, v1, v2):
        """
        Compute angle (in degrees) between two vectors.
        """
        
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1, 1)
        return np.degrees(np.arccos(cos_theta))

    def choose_edges(self, mask):
        """
        Idea is to extend line segments into lines and check intersections of all 4 lines combinations
        then check which quadrilateral is most like a square (sides similar length and angles close to 90 degrees)
        and return most square similar combinatoin of edges- places of their intersections are board corners
        """

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
        """Warp frame into canonical board coordinates."""
        return cv2.warpPerspective(frame, self.M, (self.canonical_size, self.canonical_size))


    def unwarp(self, warped_frame):
        """Map canonical image back to original frame coordinates."""
        h, w = self.frame_shape
        return cv2.warpPerspective(warped_frame, self.M_inv, (w, h))
    

    def warp_points(self, points):
        """Transform points from image space to canonical board space."""
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)
        return warped_pts
    

    def unwarp_points(self, points):
        """Transform points from canonical board space back to image space."""
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        orig_pts = cv2.perspectiveTransform(pts, self.M_inv).reshape(-1, 2)
        return orig_pts
    
    def points_inside_board(self, points_unwarped):
        """
        Check whether points lie inside the canonical board.

        Returns:
            List[bool]
        """

        if self.board is None:
            raise RuntimeError("Board not detected")
        
        points = self.warp_points(points_unwarped)
        h = w = self.canonical_size


        results = []
        for x, y in points:
            x, y = int(round(x)), int(round(y))
            results.append(0 <= x < w and 0 <= y < h)

        return results

    @property
    def ready(self):
        """True if a valid homography has been computed."""
        return self.M is not None

    def get_M(self):
        return self.M

    def get_M_inv(self):
        return self.M_inv
