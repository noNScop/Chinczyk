from collections import deque
import numpy as np

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

    def update(self, die_number, die_visible, frame_num):
        """Call every frame"""
        self.frame_num = frame_num
        self.history_bin.append(1 if die_visible else 0)
        if die_visible:
            self.history.append(die_number)
        else:
            self.history.append(-1)

        # only try to detect event if ready
        if self.ready_for_next_event:
            self.if_event = self._if_event()
            if self.if_event:
                self.ready_for_next_event = False
        else:
            # check if we can reset for next event
            neg_frac = np.mean(np.array(self.history_bin) == 0)
            if neg_frac >= self.reset_frac:
                self.ready_for_next_event = True
            self.if_event = False

    def _if_event(self):
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
                self.last_event_frame = self.frame_num
                return True

        return False
    
    def which_event(self):
        """Return the mean die value (rounded up) in recent history"""
        valid_numbers = [n for n in self.history if n > 0]
        if not valid_numbers:
            return None
        mean_val = np.mean(valid_numbers)
        return int(np.ceil(mean_val))


# from collections import deque
# import numpy as np
# from collections import Counter


# class DieThrowRecognizer():
#     def __init__(self, patience=60, min_delay=90, thresh=0.66):
#         self.history_bin = deque(maxlen=patience)  # stores 1 for die visible, 0 for not
#         self.history = deque(maxlen=patience)  # stores detecte values

#         self.min_delay = min_delay
#         self.last_event_frame = -min_delay
#         self.frame_num = 0
#         self.thresh = thresh
#         self.if_event = False

#     def update(self, die_number, die_visible, frame_num):
#         """Call every frame"""
#         self.frame_num = frame_num
#         self.history_bin.append(1 if die_visible else 0)
#         if die_visible:
#             self.history.append(die_number)
#         else:
#             self.history.append(-1)
#         self.if_event = self._if_event()

#     def _if_event(self):
#         """Return True if a throw event is detected in this frame"""
#         if len(self.history_bin) < self.history_bin.maxlen:
#             return False  # not enough frames yet

#         # split history into older half and newer half
#         hist_arr = np.array(self.history_bin)
#         half = len(hist_arr) // 2
#         older = hist_arr[:half]
#         newer = hist_arr[half:]

#         older_neg_frac = np.mean(older == 0)
#         newer_pos_frac = np.mean(newer == 1)

#         if newer_pos_frac >= self.thresh and older_neg_frac >= self.thresh:
#             # enforce min_delay between events
#             if self.frame_num - self.last_event_frame >= self.min_delay:
#                 self.last_event_frame = self.frame_num
#                 return True

#         return False
    
#     def which_event(self):
#         """Return the most common die number in the recent history (ignoring -1)"""
#         print("history", self.history)
#         valid_numbers = [n for n in self.history if n > 0]
#         if not valid_numbers:
#             return None
#         #return Counter(valid_numbers).most_common(1)[0][0]
#         mean_val = np.mean(valid_numbers)
#         return int(np.ceil(mean_val))
#         # rms_val = np.sqrt(np.mean(np.square(valid_numbers)))
#         # return int(np.ceil(rms_val))