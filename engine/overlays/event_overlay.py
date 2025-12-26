import cv2
import numpy as np
from collections import deque
import math
import random

class EventOverlay:
    """
    Casino-style animated event overlay system.
    Supports stacked effects, flashes, shakes, glows, particles.
    """

    def __init__(self):
        self.active_events = deque()
        self.global_shake = 0

    def add_event(self, text, effect_func, duration=50):
        if text is None:
            return

        # normalize to list of effects
        if not isinstance(effect_func, (list, tuple)):
            effect_func = [effect_func]

        self.active_events.append({
            'text': str(text),
            'remaining': duration,
            'duration': duration,
            'effects': effect_func
        })

    def draw(self, frame):
        frame = frame.copy()

        # global screen shake
        if self.global_shake > 0:
            dx = random.randint(-self.global_shake, self.global_shake)
            dy = random.randint(-self.global_shake, self.global_shake)
            frame = np.roll(frame, (dy, dx), axis=(0, 1))
            self.global_shake -= 1

        new_queue = deque()
        for evt in self.active_events:
            if evt['remaining'] > 0:
                for eff in evt['effects']:
                    frame = eff(
                        frame,
                        evt['text'],
                        evt['remaining'],
                        evt['duration'],
                        self
                    )
                evt['remaining'] -= 1
                new_queue.append(evt)

        self.active_events = new_queue
        return frame

    # =====================================================
    # CASINO EFFECTS
    # =====================================================

    @staticmethod
    def flash_bg(frame, text, remaining, duration, overlay):
        """Bright flashing background"""
        alpha = math.sin((duration - remaining) / duration * math.pi)
        flash = np.full_like(frame, (255, 255, 255))
        return cv2.addWeighted(frame, 1 - 0.4 * alpha, flash, 0.4 * alpha, 0)

    @staticmethod
    def pop_scale(frame, text, remaining, duration, overlay, base_scale=3.0):
        """Text pop-in with overshoot"""
        progress = (duration - remaining) / duration
        scale = base_scale * (1 + 0.6 * math.exp(-6 * progress) * math.sin(10 * progress))

        return EventOverlay._draw_center_text(
            frame, text, scale, (0, 255, 255), thickness=3
        )

    @staticmethod
    def glow_text(frame, text, remaining, duration, overlay):
        """Neon glow text"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 3.5
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = (w - tw) // 2, (h - th) // 2 + th

        for r in range(6, 0, -2):
            cv2.putText(frame, text, (x, y), font, scale, (0, 255, 255), thickness+r, cv2.LINE_AA)

        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def shake(frame, text, remaining, duration, overlay):
        """Trigger screen shake"""
        overlay.global_shake = max(overlay.global_shake, 6)
        return frame

    @staticmethod
    def fireworks(frame, text, remaining, duration, overlay):
        """Simple particle burst"""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.randint(10, 120)
            x = int(cx + math.cos(angle) * dist)
            y = int(cy + math.sin(angle) * dist)
            cv2.circle(frame, (x, y), random.randint(2, 4), (0, 255, 255), -1)

        return frame

    # =====================================================
    # PREMADE COMBOS
    # =====================================================

    @staticmethod
    def casino_win(frame, text, remaining, duration, overlay):
        overlay.global_shake = 8
        frame = EventOverlay.flash_bg(frame, text, remaining, duration, overlay)
        frame = EventOverlay.fireworks(frame, text, remaining, duration, overlay)
        frame = EventOverlay.pop_scale(frame, text, remaining, duration, overlay, base_scale=4.0)
        frame = EventOverlay.glow_text(frame, text, remaining, duration, overlay)
        return frame

    @staticmethod
    def big_popup(frame, text, remaining, duration, overlay):
        frame = EventOverlay.pop_scale(frame, text, remaining, duration, overlay)
        frame = EventOverlay.glow_text(frame, text, remaining, duration, overlay)
        return frame

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _draw_center_text(frame, text, scale, color, thickness=2):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h - th) // 2 + th

        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        return frame
