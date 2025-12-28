from state_controllers.pawn_state_controller import PawnStateController

class LeaveBaseRecognizer():
    """
    Recognizes when a player's pawn leaves the base area.

    The recognizer monitors the pawn positions and triggers an event
    whenever a pawn moves from its base (starting position) onto the board.
    """

    def __init__(self, pawn_state: PawnStateController):
        """
        Parameters
        ----------
        pawn_state : PawnStateController
            Reference to the pawn state controller that tracks pawn positions
            on the board and in base areas.
        """

        self.pawn_state = pawn_state

        # Track the number of pawns remaining at each base
        self.green_base = None
        self.blue_base = None

        # Store the player who just moved a pawn out of base
        self.player = ""

    def initialise_state(self):
        """
        Initialize the recognizer's base counts from the current pawn state.

        Should be called at the start of monitoring or before the first update.
        """

        self.green_base = self.pawn_state.green_base
        self.blue_base = self.pawn_state.blue_base

    def update(self):
        """
        Check if any pawn has left the base since the last update.

        Returns
        -------
        bool
            True if a pawn left the base for either player, False otherwise.

        Notes
        -----
        Updates `self.player` with the color of the player whose pawn
        left the base.
        """

        # Ensure there are previous positions to compare
        if self.pawn_state.previous_positions is None:
            return False
        
        # Check if blue pawn left its base (position 0)
        if 0 not in self.pawn_state.previous_positions and 0 in self.pawn_state.positions and self.pawn_state.positions[0] == "blue":
            self.player = "Blue"
            return True
        
        # Check if green pawn left its base (position 20)
        elif 20 not in self.pawn_state.previous_positions and 20 in self.pawn_state.positions and self.pawn_state.positions[20] == "green":
            self.player = "Green"
            return True
        return False