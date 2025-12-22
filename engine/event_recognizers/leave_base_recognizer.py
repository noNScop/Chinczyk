import cv2
import numpy as np

from state_controllers.pawn_state_controller import PawnStateController

class LeaveBaseRecognizer():
    """
    Recognizer for leave-base event
    Detects the decrease in number of pawns in base directly from detector, but only if all pawns are visible
    """
    def __init__(self, pawn_state: PawnStateController):
        self.pawn_state = pawn_state

        self.green_base = None
        self.blue_base = None

        self.player = ""

    def initialise_state(self):
        self.green_base = self.pawn_state.green_base
        self.blue_base = self.pawn_state.blue_base

    def update(self, player, green_pawns_visible, blue_pawns_visible):
        self.player = player

        is_event = False

        if green_pawns_visible == 4 and blue_pawns_visible == 4:
            # Initialise on the beginning
            if self.green_base is None:
                self.initialise_state()

            if self.pawn_state.green_base + 1 == self.green_base:
                is_event = True

            if self.pawn_state.blue_base + 1 == self.blue_base:
                is_event = True
            
            # Update in case the number of pawns in base increased
            self.green_base = self.pawn_state.green_base
            self.blue_base = self.pawn_state.blue_base

        return is_event