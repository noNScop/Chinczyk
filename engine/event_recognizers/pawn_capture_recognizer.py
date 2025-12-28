from state_controllers.pawn_state_controller import PawnStateController
from state_controllers.turn_state_controller import TurnStateController

class PawnCaptureRecognizer():
    def __init__(self, pawn_state: PawnStateController):
        """
        Parameters
        ----------
        pawn_state : PawnStateController
            Reference to the pawn state controller that tracks pawn positions
            on the board and in base areas.
        """

        self.pawn_state = pawn_state

        # Track current player's turn and the "prey" (opponent pawn)
        self.turn = ""
        self.prey = ""

        # Number of pawns of prey currently at base at the start of monitoring
        self.prey_at_base = float('inf')

        # Internal flags to detect capture
        self.capture_trigger = False # True if a pawn moved to a position previously occupied by another pawn
        self.base_trigger = False    # True if the prey pawn count at base increased

    def reset(self):
        """
        Reset internal state for a new turn or after detecting an event.
        Updates the turn, resets prey base count, and clears triggers.
        """

        self.turn = TurnStateController.ID_MARKER_MAPPING[self.pawn_state.turn_state.turn]

        self.prey_at_base = float('inf')

        self.capture_trigger = False
        self.base_trigger = False

    def update(self):
        """
        Check if a pawn capture event occurred in the current state.

        Returns
        -------
        bool
            True if a pawn capture was detected, False otherwise.

        Notes
        -----
        - Detects capture by checking if a pawn moved to a position
          previously occupied by an opponent pawn.
        - Confirms capture if the opponent's pawn count at base increases.
        - Resets internal state after a successful detection or when turn changes.
        """

        # Require previous positions for comparison
        if self.pawn_state.previous_positions is None:
            return False
        
        # Check if the turn changed
        if TurnStateController.ID_MARKER_MAPPING[self.pawn_state.turn_state.turn] != self.turn:
            self.reset()

            # Determine which player is the prey (opponent)
            if self.turn == "green":
                self.prey = "blue"
            else:
                self.prey = "green"

            # Record current number of prey pawns at base 
            if self.prey == "green":
                self.prey_at_base = self.pawn_state.green_base
            else:
                self.prey_at_base = self.pawn_state.blue_base
                
            return False

        # Check if any pawn moved into a position previously occupied by another pawn
        for pos in self.pawn_state.previous_positions:
            if pos in self.pawn_state.positions and self.pawn_state.previous_positions[pos] != self.pawn_state.positions[pos]:
                self.capture_trigger = True

                break

        # Check if prey pawn count at base increased
        if self.prey == "green" and self.pawn_state.green_base > self.prey_at_base:
            self.base_trigger = True
        elif self.prey == "blue" and self.pawn_state.blue_base > self.prey_at_base:
            self.base_trigger = True

        # If both triggers are True, a capture occurred
        if self.capture_trigger and self.base_trigger:
            self.reset()
            return True

        return False