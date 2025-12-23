import cv2
import numpy as np

from state_controllers.pawn_state_controller import PawnStateController

class LeaveBaseRecognizer():
    """
    Recognizer for leave-base event
    Checks if in the last state update added pawns at their base-exit positions
    """
    def __init__(self, pawn_state: PawnStateController):
        self.pawn_state = pawn_state

        self.green_base = None
        self.blue_base = None

        self.player = ""

    def initialise_state(self):
        self.green_base = self.pawn_state.green_base
        self.blue_base = self.pawn_state.blue_base

    def update(self):
        if self.pawn_state.previous_positions is None:
            return False
        
        if 0 not in self.pawn_state.previous_positions and 0 in self.pawn_state.positions and self.pawn_state.positions[0] == "blue":
            self.player = "Blue"
            return True
        elif 20 not in self.pawn_state.previous_positions and 20 in self.pawn_state.positions and self.pawn_state.positions[20] == "green":
            self.player = "Green"
            return True
        return False