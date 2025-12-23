import cv2
import numpy as np
import os
from helpers import find_main_folder
from collections import deque

class PawnStateController():    
    '''now the state controller directly detects what is visible just from one frame
    we should probably change it so when board/ pawns are not visible it remembers the state'''
    ID_MARKER_MAPPING = {0: "blue", 1:"green"}


    def __init__(self, tiles, tiles_blue, tiles_green):
        self.tiles = tiles
        self.tiles_blue = tiles_blue
        self.tiles_green = tiles_green

        self.blue_home = None # num of blue pawns in home
        self.green_home = None

        self.blue_base = None
        self.green_base = None

        self.blue_board = None
        self.green_board = None

    def update(self, green_pos_stable, blues_pos_stable, occupied_tiles_dict, board_pos_stable):
        self.update_home_info(occupied_tiles_dict)
        self.update_base_info( green_pos_stable, blues_pos_stable, board_pos_stable)
        self.update_board_info(occupied_tiles_dict )

    def update_home_info(self, occupied_tiles_dict):
        blue_home = 0
        green_home = 0
        for key in occupied_tiles_dict.keys():
            if key in self.tiles_blue.keys():
                blue_home += 1
            if key in self.tiles_green.keys():
                green_home += 1

        self.blue_home = blue_home
        self.green_home = green_home


    def update_board_info(self, occupied_tiles_dict):
        blue_board = 0
        green_board = 0
        for key in occupied_tiles_dict.keys():
            if key in self.tiles.keys():
                if occupied_tiles_dict[key] == 'blue':
                    blue_board += 1
                if occupied_tiles_dict[key] == 'green':
                    green_board +=1

        self.blue_board = blue_board
        self.green_board = green_board



    def update_base_info(self, green_pos_stable, blues_pos_stable, board_pos_stable):
        board_poly = np.asarray(board_pos_stable, dtype=np.int32)

        def is_outside_board(pt):
            # cv2.pointPolygonTest:
            #  > 0 inside
            #  = 0 on edge
            #  < 0 outside
            return cv2.pointPolygonTest(board_poly, tuple(pt), False) < 0

        self.green_base = sum(
            1 for pt in green_pos_stable if pt is not None and is_outside_board(pt)
        )

        self.blue_base = sum(
            1 for pt in blues_pos_stable if pt is not None and is_outside_board(pt)
        )
                    
    