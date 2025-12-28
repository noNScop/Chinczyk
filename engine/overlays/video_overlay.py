import cv2
import numpy as np

from detectors.die_handler import Die_handler

class VideoOverlay:
    """
    Utility class for creating visual overlays on video frames
    to visualize game state, pawns, dice, board, and event notifications.
    Each method returns a copy of the frame with the overlay applied.
    """

    @staticmethod
    def draw_green_pawn_circles(frame, pawn_centers):
        frame = frame.copy()
        for cx, cy in pawn_centers:
            cv2.circle(frame, (int(cx), int(cy)), 15, (255, 255, 51), 2)

        return frame

    @staticmethod
    def draw_blue_pawn_circles(frame, pawn_centers):
        frame = frame.copy()
        for cx, cy in pawn_centers:
            cv2.circle(frame, (int(cx), int(cy)), 15, (0, 0, 255), 2)

        return frame
    
    @staticmethod
    def draw_die(frame, pts, die_handler : Die_handler):
        frame = frame.copy()

        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        num = die_handler.get_number()

        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(
            frame,
            f"{int(num)}",
            (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        circles = die_handler.get_circes()
        if circles is not None:
            for x, y, r in circles:
                x, y, r = int(x), int(y), int(r)
                cv2.circle(frame, (x, y), r, (0, 0, 255), 2)   # circle
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1) # center
                cv2.putText(
                    frame,
                    f"r={r}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1
                )
        return frame
    
    @staticmethod
    def draw_reflection(frame, pts):
        frame = frame.copy()

        cv2.polylines(frame, [pts], True, (255, 255, 0), 2)  # cyan outline
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(
            frame,
            "reflection",
            (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        return frame

    @staticmethod
    def draw_marker(frame, pts):
        frame = frame.copy()
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)  # red outline
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(
            frame,
            "marker",
            (cx - 20, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        return frame
    
    @staticmethod
    def draw_board_boarder(frame, ordered):
        frame = frame.copy()

        if ordered is not None:
            pts = ordered.astype(int)
            # draw border
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
            labels = ["TL","TR","BR","BL"]

            for (x,y), c, l in zip(ordered, colors, labels):
                cv2.circle(frame, (int(x),int(y)), 10, c, -1)
                cv2.putText(frame, l, (int(x)+10,int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
            return frame
        
    @staticmethod
    def draw_tile_hulls(frame, internal_board, draw_alpha=0.5):
        frame = frame.copy()

        if internal_board.last_unwarped_overlay is not None:
            # Resize overlay to match frame size
            overlay_resized = cv2.resize(
                internal_board.last_unwarped_overlay,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            mask = np.any(overlay_resized != 0, axis=-1)
            frame[mask] = (
                draw_alpha * overlay_resized[mask] +
                (1 - draw_alpha) * frame[mask]
            ).astype(np.uint8)

        return frame