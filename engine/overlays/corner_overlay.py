import cv2
from state_controllers.turn_state_controller import TurnStateController
from state_controllers.pawn_state_controller import PawnStateController

'''as you wanted seperate class for overlaying info in the corner of frame'''
class CornerOverlay: 
    @staticmethod
    def draw_turn_info(frame, turn_detector: TurnStateController, padding=20):
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


    @staticmethod
    def draw_pawn_info(frame, pawn_state : PawnStateController, padding=20, y_offset=60):
        frame = frame.copy()

        lines = [
            f"Green  | home: {pawn_state.green_home}  base: {pawn_state.green_base} board: {pawn_state.blue_board}",
            f"Blue   | home: {pawn_state.blue_home}   base: {pawn_state.blue_base} board: {pawn_state.blue_board}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        line_height = 22

        # compute box size
        widths = [
            cv2.getTextSize(line, font, scale, thickness)[0][0]
            for line in lines
        ]
        box_width = max(widths) + 20
        box_height = line_height * len(lines) + 10

        top_left = (padding - 10, padding + y_offset - 20)
        bottom_right = (padding - 10 + box_width, padding + y_offset - 20 + box_height)

        # background
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)

        # draw text
        for i, line in enumerate(lines):
            y = padding + y_offset + i * line_height
            color = (0, 255, 0) if "Green" in line else (255, 0, 0)
            cv2.putText(
                frame,
                line,
                (padding, y),
                font,
                scale,
                color,
                thickness,
                cv2.LINE_AA
            )

        return frame


