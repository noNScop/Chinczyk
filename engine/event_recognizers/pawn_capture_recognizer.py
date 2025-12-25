import cv2
import numpy as np

from state_controllers.pawn_state_controller import PawnStateController
from state_controllers.turn_state_controller import TurnStateController

class PawnCaptureRecognizer():
    def __init__(self, pawn_state: PawnStateController):
        self.pawn_state = pawn_state

        self.turn = ""
        self.prey = ""

        self.prey_at_base = float('inf')

        self.capture_trigger = False
        self.base_trigger = False

    def reset(self):
        self.turn = TurnStateController.ID_MARKER_MAPPING[self.pawn_state.turn_state.turn]

        self.prey_at_base = float('inf')

        self.capture_trigger = False
        self.base_trigger = False

    def update(self):
        if self.pawn_state.previous_positions is None:
            return False
        
        if TurnStateController.ID_MARKER_MAPPING[self.pawn_state.turn_state.turn] != self.turn:
            self.reset()

            if self.turn == "green":
                self.prey = "blue"
            else:
                self.prey = "green"
                
            if self.prey == "green":
                self.prey_at_base = self.pawn_state.green_base
            else:
                self.prey_at_base = self.pawn_state.blue_base
                
            return False

        for pos in self.pawn_state.previous_positions:
            if pos in self.pawn_state.positions and self.pawn_state.previous_positions[pos] != self.pawn_state.positions[pos]:
                self.capture_trigger = True

                break

        if self.prey == "green" and self.pawn_state.green_base > self.prey_at_base:
            self.base_trigger = True
        elif self.prey == "blue" and self.pawn_state.blue_base > self.prey_at_base:
            self.base_trigger = True

        if self.capture_trigger and self.base_trigger:
            self.reset()
            return True

        return False