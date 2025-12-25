from state_controllers.turn_state_controller import TurnStateController

class PawnStateController():    
    """
    PawnStateController maintains positions from the current and previous state
    (home / board / base) derived from per-frame detections.
    """

    # Nie widzę żadnych korzyści z tego mapowania, a wszędzie gdzie trzeba tego użyć kod staje się nie czytelny
    ID_MARKER_MAPPING = {0: "blue", 1:"green"}


    def __init__(self, tiles, tiles_blue, tiles_green, turn_state, board_detector):
        # Reference to the current turn state (used to gate position updates)
        self.turn_state = turn_state

        # Used to determine if a pawn is on the board or in the base
        self.board_detector = board_detector

        # Accepted pawn-to-tile mapping for the current stable state
        self.positions = None

        # Previously accepted pawn-to-tile mapping (used for event detection)
        self.previous_positions = None

        # Tile definitions (global board + per-color home tiles)
        self.tiles = tiles
        self.tiles_blue = tiles_blue
        self.tiles_green = tiles_green

        # Aggregated pawn counts by region
        self.blue_home = None
        self.green_home = None
        self.blue_base = None
        self.green_base = None
        self.blue_board = None
        self.green_board = None

    def update(self, green_pos_stable, blues_pos_stable, occupied_tiles_dict):
        """
        Updates the pawn state based on the current frame.

        The update proceeds in three stages:
        1. Accept or reject a new discrete pawn-to-tile mapping.
        2. Recompute home and board occupancy from the accepted mapping.
        3. Infer base occupancy from the number of visible pawns.

        Updates are only applied when all pawns of both colors are visible.
        """
        self.previous_positions = None

        self.update_positions(green_pos_stable, blues_pos_stable, occupied_tiles_dict)
        self.update_home_info()
        self.update_base_info(green_pos_stable, blues_pos_stable)
        self.update_board_info()

    def update_positions(self, green_pos_stable, blues_pos_stable, occupied_tiles_dict):
        # Ensure full visibility of all pawns before considering an update
        if len(green_pos_stable) == 4 and len(blues_pos_stable) == 4:
            
            self.previous_positions = self.positions

            # Accept initial state or states with increased or unchanged coverage
            if self.positions is None or len(self.positions) <= len(occupied_tiles_dict):
                self.positions = occupied_tiles_dict
            # Accept states with decreased coverage only if pawn was captured
            else:
                for old_p in self.positions:
                    for new_p in occupied_tiles_dict:
                        if old_p == new_p and self.positions[old_p] != occupied_tiles_dict[new_p]:
                            self.positions = occupied_tiles_dict

    def update_home_info(self):
        """
        Counts how many pawns of each color occupy home tiles,
        based on the currently accepted pawn-to-tile mapping.
        """
        if self.positions is None: return

        blue_home = 0
        green_home = 0
        for key in self.positions.keys():
            if key in self.tiles_blue.keys():
                blue_home += 1
            if key in self.tiles_green.keys():
                green_home += 1

        self.blue_home = blue_home
        self.green_home = green_home

    def update_board_info(self):
        """
        Counts how many pawns of each color occupy regular board tiles,
        excluding base and home positions.
        """
        if self.positions is None: return

        blue_board = 0
        green_board = 0
        for key in self.positions.keys():
            if key in self.tiles.keys():
                if self.positions[key] == 'blue':
                    blue_board += 1
                if self.positions[key] == 'green':
                    green_board += 1

        self.blue_board = blue_board
        self.green_board = green_board

    def update_base_info(self, green_pos_stable, blues_pos_stable):
        """
        Infers base occupancy by checking whether each pawn lies inside the board region.

        Preconditions:
        - Exactly 4 stable pawn positions per color must be available.
        - The board detector must have a valid board mask.

        Logic:
        - A pawn is considered "in base" if its position lies outside the detected board mask.
        - Base count equals the number of pawns outside the board region.

        This approach avoids false base classification caused by
        transient detection dropouts, assuming stability filtering
        has already been applied upstream.
        """
        if len(green_pos_stable) == 4 and len(blues_pos_stable) == 4:
            self.green_base = 4
            for res in self.board_detector.points_inside_board(green_pos_stable):
                self.green_base -= res

            self.blue_base = 4
            for res in self.board_detector.points_inside_board(blues_pos_stable):
                self.blue_base -= res