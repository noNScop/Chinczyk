class MoveSuggester:
    """
    Suggests possible moves for pawns based on the current game state and die roll.

    Attributes
    ----------
    green_route : list
        Sequence of tile IDs representing the green player's path.
    blue_route : list
        Sequence of tile IDs representing the blue player's path.
    die_number : int or None
        Last die roll used for generating suggestions.
    suggestions : list or None
        Currently generated suggested tiles for moves.
    if_suggest : bool or None
        Flag indicating whether suggestion mode is active.
    new_suggestion : bool or None
        Flag indicating if a new die roll has triggered fresh suggestions.
    """

    def __init__(self):
        # Initialize routes for each player
        self.green_route = self.init_green()
        self.blue_route = self.init_blue()

        # Die roll and suggestion-related state
        self.die_number = None
        self.if_suggest = None
        self.suggestions = None
        self.new_suggestion = None


    def suggest_moves(self, occupied_tiles : dict, if_green_turn : bool, blue_base : int, green_base: int):
        """
        Suggest possible tiles a pawn can move to given the current game state and die roll.

        Parameters
        ----------
        occupied_tiles : dict
            Dictionary mapping tile ID to occupying player ('green' or 'blue').
        if_green_turn : bool
            True if it's green player's turn, False otherwise.
        blue_base : int
            Number of pawns remaining in blue base.
        green_base : int
            Number of pawns remaining in green base.

        Returns
        -------
        list
            Tile IDs where pawns can legally move.
        """

        suggestions = []

        if self.die_number == None:
            return [] # No die roll, no suggestions
        
        # Determine active player
        if not if_green_turn:
            val = 'blue' 
            route = self.blue_route
            base_num = blue_base
        else:
            val = 'green'
            route = self.green_route
            base_num = green_base

        # Check if pawn can leave base on a roll of 6
        if self.die_number == 6 and route[0] not in occupied_tiles and base_num > 0:
            suggestions.append(route[0]) # you can put your pawn out of base
        
        # Check moves for pawns already on the board
        for key, player in occupied_tiles.items():
            if player != val:
                continue
            num_on_route = route.index(key) # which tile on route of pawn it is
            if num_on_route + self.die_number >= len(route):
                continue # player wants to move too far
            possible_tile = route[num_on_route + self.die_number] # tile id on which you can possibly stand on
            if possible_tile not in occupied_tiles.keys() or occupied_tiles[possible_tile] != val: # if nobody stands on the tile or enemy stands on the tile
                suggestions.append(possible_tile)

        return suggestions

    def start_suggestion(self, die_number):
        """
        Triggered by die throw recognizer when a die roll is detected.
        Enables suggestion mode and sets die number.
        """

        self.if_suggest = True
        self.new_suggestion = True
        self.die_number = die_number


    def stop_suggestion(self):
        """
        Stop suggestions when the turn changes.
        Clears die number and current suggestions.
        """

        self.if_suggest = False
        self.die_number = None
        self.suggestions = None


    @staticmethod
    def init_blue():
        """Initialize the sequence of tiles for the blue player."""
        return [i for i in range(44)]
    

    @staticmethod
    def init_green():
        """Initialize the sequence of tiles for the green player."""
        track = [i for i in range(20,40)]
        track += [i for i in range(20)]
        track += [44, 45, 46, 47]
        return track
    

    def get_if_suggest(self):
        """Return whether suggestion mode is currently active."""
        return self.if_suggest
    

    def make_suggestions(self, occupied_tiles : dict, if_green_turn : bool, blue_base : int,  green_base: int ):
        """
        Generate suggestions if they have not been generated yet or
        a new die roll has occurred.

        Updates `self.suggestions` with suggested tiles.
        """

        if self.suggestions == None or self.new_suggestion == True:
            self.new_suggestion = False
            self.suggestions = self.suggest_moves( occupied_tiles , if_green_turn, blue_base, green_base)


    def get_suggestions(self):
        return self.suggestions