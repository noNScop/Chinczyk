from state_controllers.pawn_state_controller import PawnStateController

class EnterHomeRecognizer():
    """
    Recognizes when a player's pawn enters the home area.

    Uses the PawnStateController to monitor the number of pawns
    currently in the home area for each player. An event is triggered
    whenever the count increases compared to the previous frame.
    """

    def __init__(self, pawn_state: PawnStateController):
        """
        Parameters
        ----------
        pawn_state : PawnStateController
            Reference to the pawn state controller that maintains
            current positions and counts of pawns.
        """

        self.pawn_state = pawn_state

        # Track previously observed number of pawns at home
        self.green_at_home = self.pawn_state.green_home
        self.blue_at_home = self.pawn_state.blue_home

        # Store the last player whose pawn entered home
        self.last_entered = ""


    def update(self, turn):
        """
        Check if a pawn has entered home during the current turn.

        Parameters
        ----------
        turn : str
            Current player's turn ("green" or "blue").

        Returns
        -------
        bool
            True if a pawn entered home for the current player
            since the last update, False otherwise.

        Notes
        -----
        Updates `self.last_entered` with the player's color
        when a pawn enters home.
        """

        if turn == "green":
            # If we already have a reference count for green pawns at home
            if self.green_at_home is not None:
                if self.pawn_state.green_home > self.green_at_home:
                    # A pawn entered home
                    self.green_at_home = self.pawn_state.green_home
                    self.last_entered = "Green"
                    return True
            else:
                # Initialize count if it was None
                self.green_at_home = self.pawn_state.green_home
        else: # turn == "blue"
            if self.blue_at_home is not None:
                if self.pawn_state.blue_home > self.blue_at_home:
                    self.blue_at_home = self.pawn_state.blue_home
                    self.last_entered = "Blue"
                    return True
            else:
                self.blue_at_home = self.pawn_state.blue_home
        
        return False