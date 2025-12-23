import cv2
import numpy as np

from state_controllers.pawn_state_controller import PawnStateController

class EnterHomeRecognizer():
    """
    Recognizer for enter-home event
    Detects the change in number of pawns in home
    To filter out false positives:
        - check if it is the turn of player with entering pawn
        - store the maximum number of pawns in base detected and fire event only if it increases
        - check if any pawn was in a reachable range to the base shortly before entering (Not implemented)

    Ten ostatni filter cięko było by mi na ten moment zrobić bo jeszcze dokońca nie wiem jak są ponumerowane pola,
    ale najwyżej później do tego wrócę
    """
    
    def __init__(self, pawn_state: PawnStateController):
        self.pawn_state = pawn_state

        self.green_at_home = self.pawn_state.green_home
        self.blue_at_home = self.pawn_state.blue_home

        self.last_entered = ""


    def update(self, turn):
        if turn == "green" :
            if self.green_at_home is not None:
                if self.pawn_state.green_home > self.green_at_home:
                    self.green_at_home = self.pawn_state.green_home
                    self.last_entered = "Green"
                    return True
        else:
            if self.blue_at_home is not None:
                if self.pawn_state.blue_home > self.blue_at_home:
                    self.blue_at_home = self.pawn_state.blue_home
                    self.last_entered = "Blue"
                    return True
        
        return False