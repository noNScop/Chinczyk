from state_controllers.pawn_state_controller import PawnStateController

class WinGameRecognizer:
    """
    Recognizes if the game has ended.
    The game ends when all pawns of one color are in their home area.
    """

    def __init__(self, pawn_state: PawnStateController, pawns_per_player=4):
        self.pawn_state = pawn_state
        self.pawns_per_player = pawns_per_player
        self.winner = None  # "green", "blue", or None
        self.game_over = False


    def update(self):
        if self.pawn_state.green_home == self.pawns_per_player:
            self.winner = "green"
            self.game_over = True
        elif self.pawn_state.blue_home == self.pawns_per_player:
            self.winner = "blue"
            self.game_over = True
        else:
            self.winner = None
            self.game_over = False
        return self.game_over


    def get_winner(self):
        return self.winner
