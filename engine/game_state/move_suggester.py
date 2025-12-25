import numpy as np

class MoveSuggester:
    def __init__(self):
        
        self.green_route = self.init_green()
        self.blue_route = self.init_blue()

        self.die_number = None

        self.suggestions = None
        self.if_suggest= None

    def suggest_moves(self, occupied_tiles : dict, if_green_turn : bool, blue_base : int, green_base: int,  ):# occupied_tiles example {47: 'green', 16: 'green', 19: 'green', 38: 'blue', 42: 'blue'}

        """
        Returns list: all suggested tiles
        """
        suggestions = []

        if self.die_number == None:
            return []
        
        if not if_green_turn:
            val = 'blue' 
            route = self.blue_route
            base_num = blue_base
        else:
            val = 'green'
            route = self.green_route
            base_num = green_base

        if self.die_number == 6 and route[0] not in occupied_tiles and base_num > 0:
            suggestions.append(route[0]) # you can put your pawn out of base
        
        for key, player in occupied_tiles.items():
            if player != val:
                continue
            num_on_route = route.index(key) # which tile on route of pawn it is
            if num_on_route + self.die_number >= len(route):
                continue # player want to move too far
            possible_tile = route[num_on_route + self.die_number] # tile id on which you can possibly stand on
            if possible_tile not in occupied_tiles.keys() or occupied_tiles[possible_tile] != val: # if nobody stands on the tile or enemy stands on the tile
                suggestions.append(possible_tile)

        return suggestions

    def start_suggestion(self, die_number):  #launched by die_throw recognizer when throw recognized
        self.if_suggest = True
        self.die_number = die_number
        print("starting suggestions!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    def stop_suggestion(self): # launched by turn_state_controller when turn changes
        print("stopping suggestions")
        self.if_suggest = False
        self.die_number = None
        self.suggestions = None

    @staticmethod
    def init_blue():
        return [i for i in range(44)]
    
    @staticmethod
    def init_green():
        track = [i for i in range(20,40)]
        track += [i for i in range(20)]
        track += [44, 45, 46, 47]
        return track
    
    def get_if_suggest(self):
        return self.if_suggest
    
    def make_suggestions(self, occupied_tiles : dict, if_green_turn : bool, blue_base : int,  green_base: int ): # launched in main
        print("making suggestions##################################")
        if self.suggestions == None:
            self.suggestions = self.suggest_moves( occupied_tiles , if_green_turn, blue_base, green_base)

    def get_suggestions(self):
        print("suuuuuuuuuuuuuuugestoins", self.suggestions)
        return self.suggestions


