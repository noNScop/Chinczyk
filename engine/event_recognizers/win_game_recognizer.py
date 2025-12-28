from state_controllers.pawn_state_controller import PawnStateController

class WinGameRecognizer:
    """
    Recognizes if the game has ended.

    The game is considered over when all pawns of a player have reached
    their home area. Tracks the winner and provides a method to check
    the game status.
    """

    def __init__(self, pawn_state: PawnStateController, pawns_per_player=4):
        """
        Parameters
        ----------
        pawn_state : PawnStateController
            Reference to the pawn state controller that tracks pawns on the board
            and at home for each player.
        pawns_per_player : int, optional
            Number of pawns each player has, by default 4
        """

        self.pawn_state = pawn_state
        self.pawns_per_player = pawns_per_player
        self.winner = None  # "green", "blue", or None if no winner yet
        self.game_over = False


    def update(self):
        """
        Check if the game has ended.

        Returns
        -------
        bool
            True if the game is over, False otherwise.

        Notes
        -----
        - Updates the winner attribute if a player has all pawns at home.
        - Sets game_over flag to True if a winner is found.
        """

        if self.pawn_state.green_home == self.pawns_per_player and self.winner is None:
            self.winner = "green"
            self.game_over = True
        elif self.pawn_state.blue_home == self.pawns_per_player and self.winner is None:
            self.winner = "blue"
            self.game_over = True
        else:
            self.game_over = False
        return self.game_over


    def get_winner(self):
        return self.winner
