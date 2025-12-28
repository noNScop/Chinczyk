import cv2
from state_controllers.turn_state_controller import TurnStateController
from state_controllers.pawn_state_controller import PawnStateController

class CornerOverlay:
    """
    Utility class to draw overlay information on the video frame
    about the current turn and pawn state.
    """

    @staticmethod
    def draw_turn_info(frame, turn_detector: TurnStateController, padding=20):
        """
        Draw the current player's turn information in the top-left corner.

        Parameters
        ----------
        frame : np.ndarray
            The BGR image/frame to draw on.
        turn_detector : TurnStateController
            Instance that keeps track of whose turn it is.
        padding : int, optional
            Space from the frame edge (default is 20 pixels).

        Returns
        -------
        np.ndarray
            Copy of the frame with the turn information overlay.
        """

        frame = frame.copy()

        # Determine text and color based on current turn
        if turn_detector.turn is None:
            text = "Turn: unknown"
            color = (120, 120, 120) 
        else:
            player = turn_detector.ID_MARKER_MAPPING[turn_detector.turn]
            text = f"Turn: {player}"
            color = (0, 255, 0) if player == "green" else (255, 0, 0)

        # Calculate size of text for background rectangle
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (padding - 10, padding - th - 10),
            (padding + tw + 10, padding + 5),
            (0, 0, 0), -1)

        # Draw text over rectangle
        cv2.putText(
            frame, text, (padding, padding),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        return frame


    @staticmethod
    def draw_pawn_info(frame, pawn_state : PawnStateController, padding=20, y_offset=60):
        """
        Draw detailed pawn information for both players (green and blue).

        Shows number of pawns at home, in base, and on the board.

        Parameters
        ----------
        frame : np.ndarray
            The BGR image/frame to draw on.
        pawn_state : PawnStateController
            Instance tracking current pawn positions for each player.
        padding : int, optional
            Horizontal space from the left frame edge (default 20 px).
        y_offset : int, optional
            Vertical offset from the top for the info box (default 60 px).

        Returns
        -------
        np.ndarray
            Copy of the frame with pawn info overlay.
        """

        frame = frame.copy()

        # Prepare text lines for each player
        lines = [
            f"Green  | home: {pawn_state.green_home}  base: {pawn_state.green_base} board: {pawn_state.green_board}",
            f"Blue   | home: {pawn_state.blue_home}   base: {pawn_state.blue_base} board: {pawn_state.blue_board}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        line_height = 22

        # Determine width and height of background rectangle
        widths = [
            cv2.getTextSize(line, font, scale, thickness)[0][0]
            for line in lines
        ]
        box_width = max(widths) + 20
        box_height = line_height * len(lines) + 10

        top_left = (padding - 10, padding + y_offset - 20)
        bottom_right = (padding - 10 + box_width, padding + y_offset - 20 + box_height)

        # Draw background rectangle for readability
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)

        # Draw each line of text with appropriate color
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
