import cv2
from collections import deque

class EventOverlay:
    """
    Display short-lived event messages with different visual effects.
    Each event type has its own display function.
    """

    def __init__(self):
        # Active events: each element is dict with keys:
        # 'text', 'type', 'remaining', 'effect_func'
        self.active_events = deque()

    def add_event(self, text, effect_func, duration=50):
        """Add a new event to display queue"""
        if text is not None:
            self.active_events.append({
                'text': str(text),
                'remaining': duration,
                'effect_func': effect_func
            })

    def draw(self, frame):
        """Draw all active events with their effect"""
        frame = frame.copy()
        new_queue = deque()

        for evt in self.active_events:
            if evt['remaining'] > 0:
                # Call effect function to render this event
                frame = evt['effect_func'](frame, evt['text'], evt['remaining'])
                evt['remaining'] -= 1
                new_queue.append(evt)

        self.active_events = new_queue
        return frame

    # ----------------------
    # Example effect functions
    # ----------------------

    @staticmethod
    def fade_center(frame, text, remaining, duration=50, font_scale=4.0, thickness=2):
        """
        Text in center, fades out as remaining decreases
        """
        h, w = frame.shape[:2]
        alpha = remaining / duration
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - tw) // 2
        y = (h - th) // 2

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x-20, y-20), (x+tw+20, y+th+20), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.6*alpha, frame, 1-0.6*alpha, 0)

        # Text
        color = (0, int(255*alpha), int(255*alpha))  # cyan fade
        cv2.putText(frame, text, (x, y+th), font, font_scale, color, thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def bounce_text(frame, text, remaining, duration=50, font_scale=4.0, thickness=2):
        """
        Text in center that bounces (scales up and down)
        """
        import math
        h, w = frame.shape[:2]
        progress = (duration - remaining) / duration
        scale = font_scale * (1.0 + 0.5*math.sin(progress*math.pi*2))  # oscillate
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h - th) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x-15, y-15), (x+tw+15, y+th+15), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        cv2.putText(frame, text, (x, y+th), font, scale, (0,255,255), thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def slide_up(frame, text, remaining, duration=50, font_scale=4.0, thickness=2):
        """
        Text slides from bottom to center
        """
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        progress = (duration - remaining) / duration
        y = int(h - (h//2 + th) * progress)
        x = (w - tw) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x-10, y-10), (x+tw+10, y+th+10), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.putText(frame, text, (x, y+th), font, font_scale, (0,255,128), thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def shake_text(frame, text, remaining, duration=50, font_scale=4.0, thickness=2):
        """
        Text shakes randomly around center
        """
        import random
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        x = (w - tw) // 2 + random.randint(-5,5)
        y = (h - th) // 2 + random.randint(-5,5)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x-15, y-15), (x+tw+15, y+th+15), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.putText(frame, text, (x, y+th), font, font_scale, (0,255,255), thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def outline_text(frame, text, remaining, duration=50, font_scale=4.0, thickness=2):
        """
        Text with a glowing outline effect
        """
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - tw) // 2
        y = (h - th) // 2

        # Draw outline
        for dx in [-2,0,2]:
            for dy in [-2,0,2]:
                if dx==0 and dy==0: continue
                cv2.putText(frame, text, (x+dx, y+th+dy), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        # Main text
        cv2.putText(frame, text, (x, y+th), font, font_scale, (0,255,255), thickness, cv2.LINE_AA)
        return frame
