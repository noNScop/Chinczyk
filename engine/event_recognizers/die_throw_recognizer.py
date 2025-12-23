from collections import deque
import numpy as np
from collections import Counter
import cv2


class DieThrowRecognizer:# new chat version i did not check it but works better than old one
    def __init__(self, patience=60, min_delay=90, thresh=0.66, reset_frac=0.95):
        """
        patience : number of frames to keep in history
        min_delay : minimum frames between events
        thresh : fraction of frames needed to detect a throw
        reset_frac : fraction of frames that must be negative before next event
        """
        self.history_bin = deque(maxlen=patience)  # 1 if die visible, 0 if not
        self.history = deque(maxlen=patience)      # detected die values
        self.min_delay = min_delay
        self.last_event_frame = -min_delay
        self.frame_num = 0
        self.thresh = thresh
        self.reset_frac = reset_frac
        self.if_event = False
        self.ready_for_next_event = True  # new flag to handle reset

        self.prev_detected_die_pos_event = None
        self.prev_M = None
        self.detected_die_pos_event = None
        self.M = None

        self.detected_die_pos_last = None

        self.dist_threshold = 50 # TO TUNE

    def update(self, die_number, if_die_visible, frame_num, M, all_die_pos, board_poly):# na razie jest prowizorka i może być kilka kości wykrytych all_die_pos is list
        """Call every frame"""
        if len(all_die_pos)>0:
            self.detected_die_pos_last = all_die_pos[0]# primitive
        # else:
        #     self.detected_die_pos_cur = None
            # else:
        #raise Exception("Sorry, no die detected but event detected, sth is no yes")
            
        print("die cur", self.detected_die_pos_last)
        self.frame_num = frame_num
        self.history_bin.append(1 if if_die_visible else 0)
        if if_die_visible:
            self.history.append(die_number)
        else:
            self.history.append(-1)

        # only try to detect event if ready
        if self.ready_for_next_event:
            self.if_event = self._if_event(board_poly)
            if self.if_event:
                self.ready_for_next_event = False
        else:
            # check if we can reset for next event
            neg_frac = np.mean(np.array(self.history_bin) == 0)
            if neg_frac >= self.reset_frac:
                self.ready_for_next_event = True
            self.if_event = False
        
        if self.if_event:
            self.prev_detected_die_pos_event = self.detected_die_pos_event
            self.detected_die_pos_event = self.detected_die_pos_last
            
            self.prev_M = self.M
            self.M = M
            if self._cancel_event():
                self.if_event = False
        

    def _if_event(self, board_poly):
        """Return True if a throw event is detected in this frame"""
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
        """Return the mean die value (rounded up) in recent history"""
        print("history", self.history)

        valid_numbers = [n for n in self.history if n > 0]
        if not valid_numbers:
            return None
        return Counter(valid_numbers).most_common(1)[0][0]


    def _cancel_event(self): # if new event is percived we check distance of new die recognized to old die; if it is too small then event is cancelled(die was just not recognized for some frames)
        print("points:", self.detected_die_pos_event)

        if (
            self.detected_die_pos_event is None or
            self.prev_detected_die_pos_event is None or
            self.M is None or
            self.prev_M is None
        ):
            return False 
    
        warped_die_pos = self.warp_points(self.detected_die_pos_event, self.M)
        prev_warped_die_pos = self.warp_points(self.prev_detected_die_pos_event, self.prev_M)

        print("detected", self.detected_die_pos_event, self.prev_detected_die_pos_event)
        print("warped", warped_die_pos, prev_warped_die_pos)

        dist = self.calculate_dist(warped_die_pos, prev_warped_die_pos)
        print(dist, "dist")

        if dist < self.dist_threshold:
            return True
        else:
            return False
        
    def if_detected_on_board(self, board_poly):
        def is_inside_board(pt):
            pt = (int(pt[0]), int(pt[1]))

            print(pt, 'pt')
            # cv2.pointPolygonTest:
            #  > 0 inside
            #  = 0 on edge
            #  < 0 outside
            return cv2.pointPolygonTest(board_poly, tuple(pt), False) > 0

        die = self.detected_die_pos_last
        num_board = sum(1 for pt in die if pt is not None and is_inside_board(pt))
        if num_board > 0:
            return True


    @staticmethod
    def calculate_dist(marker_detected, marker_real): # can be moved t ohelpers?
        """
        Returns the sum of distances from each detected marker point
        to the closest point in the real marker template.
        """
        marker_detected = np.asarray(marker_detected, dtype=np.float32)
        marker_real = np.asarray(marker_real, dtype=np.float32)

        total_dist = 0.0

        for px, py in marker_detected:
            dists = np.linalg.norm(marker_real - np.array([px, py]), axis=1)
            total_dist += np.min(dists)

        return total_dist
    
    @staticmethod # can be moved to helpers?
    def warp_points(points, M):
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        return warped_pts
