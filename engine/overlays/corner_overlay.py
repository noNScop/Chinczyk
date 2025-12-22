import cv2
from detectors.turn_detector import TurnDetector

'''as you wanted seperate class for overlaying info in the corner of frame'''
class CornerOverlay: 
    @staticmethod
    def draw_turn_info(frame, turn_detector: TurnDetector, padding=20):
        frame = frame.copy()

        if turn_detector.turn is None:
            text = "Turn: unknown"
            color = (120, 120, 120) 
        else:
            player = turn_detector.ID_MARKER_MAPPING[turn_detector.turn]
            text = f"Turn: {player}"

            if player == "green":
                color = (0, 255, 0)
            else:  
                color = (255, 0, 0)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        cv2.rectangle(
            frame,
            (padding - 10, padding - th - 10),
            (padding + tw + 10, padding + 5),
            (0, 0, 0), -1)

        cv2.putText(
            frame, text, (padding, padding),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        return frame


