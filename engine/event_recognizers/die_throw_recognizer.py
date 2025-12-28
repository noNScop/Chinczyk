import cv2
import numpy as np
from collections import deque
from collections import Counter

from game_state.move_suggester import MoveSuggester


class DieThrowRecognizer:
    """
    Event detector for dice throws.

    The detector operates on a sliding window of frames and identifies
    a throw event as a transition:
        - die NOT visible  → die visible

    Additional safeguards:
    - minimum delay between events
    - reset condition requiring a sufficient number of negative frames
    - spatial consistency check to cancel false events
    """
    
    def __init__(self, move_suggester: MoveSuggester, patience=60, min_delay=90, thresh=0.66, reset_frac=0.95):
        """
        Parameters
        ----------
        patience : int
            Length of the sliding window (in frames)

        min_delay : int
            Minimum number of frames between two consecutive events

        thresh : float
            Fraction of frames required to trigger an event

        reset_frac : float
            Fraction of negative (no-die) frames required before allowing
            detection of a new event
        """
       
        # Binary history: 1 → die visible, 0 → no die
        self.history_bin = deque(maxlen=patience)
       
        # Detected die values per frame (-1 if not visible)
        self.history = deque(maxlen=patience)
       
        self.min_delay = min_delay
        self.last_event_frame = -min_delay
        self.frame_num = 0

        self.thresh = thresh
        self.reset_frac = reset_frac

        self.if_event = False
        self.ready_for_next_event = True

        # Spatial validation of consecutive events
        self.prev_detected_die_pos_event = None
        self.detected_die_pos_event = None
        
        self.M = None
        self.prev_M = None

        # Most recent die position (used for board check)
        self.detected_die_pos_last = None

        # Distance threshold in warped board space
        self.dist_threshold = 50 # to be tuned empirically

        self.move_suggester = move_suggester


    def update(self, die_number, if_die_visible, frame_num, M, all_die_pos, board_poly):
        """
        Update the recognizer state for a single frame.

        This method:
        - updates temporal history
        - detects throw events
        - validates spatial consistency
        - triggers move suggestions on valid events
        """

        # Store most recent detected die (primitive: first detection only)
        if len(all_die_pos)>0:
            self.detected_die_pos_last = all_die_pos[0]
            
        self.frame_num = frame_num

        # Update temporal history
        self.history_bin.append(1 if if_die_visible else 0)
        self.history.append(die_number if if_die_visible else -1)

        # Event detection logic
        if self.ready_for_next_event:
            self.if_event = self._if_event(board_poly)
            if self.if_event:
                self.ready_for_next_event = False
        else:
            # Reset condition: sufficient number of negative frames
            neg_frac = np.mean(np.array(self.history_bin) == 0)
            if neg_frac >= self.reset_frac:
                self.ready_for_next_event = True
            self.if_event = False
        
        # Event post-processing
        if self.if_event:
            self.prev_detected_die_pos_event = self.detected_die_pos_event
            self.detected_die_pos_event = self.detected_die_pos_last
            
            self.prev_M = self.M
            self.M = M

            # Cancel false events caused by temporary detection loss
            if self._cancel_event():
                self.if_event = False
            else:
                number = self.which_event()
                self.move_suggester.start_suggestion(number)
        

    def _if_event(self, board_poly):
        """
        Detect a throw event based on temporal statistics.

        An event is detected if:
        - earlier half of history mostly contains no-die frames
        - later half mostly contains die-visible frames
        - sufficient time passed since last event
        - die is not already on the board
        """

        if len(self.history_bin) < self.history_bin.maxlen:
            return False  # not enough frames yet

        hist_arr = np.array(self.history_bin)
        half = len(hist_arr) // 2

        older = hist_arr[:half]
        newer = hist_arr[half:]

        older_neg_frac = np.mean(older == 0)
        newer_pos_frac = np.mean(newer == 1)

        if newer_pos_frac >= self.thresh and older_neg_frac >= self.thresh:
            # enforce min_delay between events
            if self.frame_num - self.last_event_frame >= self.min_delay:
                if not self.if_detected_on_board(board_poly):  # die not on board
                    self.last_event_frame = self.frame_num
                    return True

        return False
    
    def which_event(self):
        """
        Determine the die value for the detected throw.

        Uses majority voting over recent valid detections.
        """

        valid_numbers = [n for n in self.history if n > 0]
        if not valid_numbers:
            return None
        return Counter(valid_numbers).most_common(1)[0][0]


    def _cancel_event(self):
        """
        Cancel an event if the new die position is too close
        to the previous one (indicating a missed detection rather
        than a new throw).
        """

        if (
            self.detected_die_pos_event is None 
            or self.prev_detected_die_pos_event is None 
            or self.M is None 
            or self.prev_M is None
        ):
            return False 
    
        warped_die_pos = self.warp_points(self.detected_die_pos_event, self.M)
        prev_warped_die_pos = self.warp_points(self.prev_detected_die_pos_event, self.prev_M)

        dist = self.calculate_dist(warped_die_pos, prev_warped_die_pos)
        return dist < self.dist_threshold
        
    def if_detected_on_board(self, board_poly):
        """
        Check whether the detected die lies on the board area.

        Used to suppress false events caused by stationary dice.
        """

        def is_inside_board(pt):
            pt = (int(pt[0]), int(pt[1]))
            # cv2.pointPolygonTest:
            # > 0 → inside
            # = 0 → on edge
            # < 0 → outside
            return cv2.pointPolygonTest(board_poly, tuple(pt), False) > 0

        die = self.detected_die_pos_last
        if die is None:
            return False
        
        return any(is_inside_board(pt) for pt in die if pt is not None)


    @staticmethod
    def calculate_dist(marker_detected, marker_real): # can be moved t ohelpers?
        """
        Compute total distance between two point sets by matching
        each detected point to the nearest reference point.
        """

        marker_detected = np.asarray(marker_detected, dtype=np.float32)
        marker_real = np.asarray(marker_real, dtype=np.float32)

        total_dist = 0.0

        for px, py in marker_detected:
            dists = np.linalg.norm(marker_real - np.array([px, py]), axis=1)
            total_dist += np.min(dists)

        return total_dist
    
    @staticmethod
    def warp_points(points, M):
        """
        Apply perspective transform to a list of 2D points.
        """
        
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        return warped_pts
