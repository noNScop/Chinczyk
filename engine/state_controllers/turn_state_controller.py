import cv2
import numpy as np
import os
from helpers import find_main_folder
from collections import deque
from game_state.move_suggester import MoveSuggester

class TurnStateController():    
    ID_MARKER_MAPPING = {0: "blue", 1:"green"}


    def __init__(self, marker_tiles, move_suggester: MoveSuggester):
        self.marker_tiles = marker_tiles
        self.markers_all = [] # on some frames more than 1 marker are recognized
        self.turn = None

        self.sim_history = deque(maxlen=30)  # each entry = [sim_blue, sim_green]
        self.move_suggerster = move_suggester

    def reset_internal_markers(self):
        self.markers_all = []

    def add_marker(self, detected_marker):
        self.markers_all.append(detected_marker)

    @staticmethod
    def warp_points(points, M):
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        return warped_pts

    def decide_on_turn(self, M):
        #print("turn", self.turn)
        if(len(self.markers_all)) == 0:
            return
        
        similarities = self.calculate_similarities(M) # real example similarities [[np.float32(0.00045678572), np.float32(0.020638706)]]
        sim_blue = np.max(similarities[:, 0])
        sim_green = np.max(similarities[:, 1])
        self.sim_history.append([sim_blue, sim_green])

        self.reset_internal_markers()
        turn = self._decide_from_memory()
        if turn != self.turn:
            self.move_suggerster.stop_suggestion()
        self.turn = turn

    def _decide_from_memory(self):
        sims = np.array(self.sim_history)

        total_blue = sims[:, 0].sum()
        total_green = sims[:, 1].sum()

        if total_blue > total_green:
            turn = 0
        else:
            turn = 1
        return turn

        # if max(total_blue, total_green) < 1e-3:
        #     return  # not enough evidence

        # ratio = max(total_blue, total_green) / (min(total_blue, total_green) + 1e-7)

        # if ratio > 2.0:  
        #     self.turn = 0 if total_blue > total_green else 1


    def calculate_similarities(self, M):
        similarities = [[0 for _ in range(2)] for _ in range(len(self.markers_all)) ]
        for i, marker_detected in enumerate(self.markers_all):
            warped_marker_detected = self.warp_points(marker_detected, M)
            for key in self.marker_tiles:
                marker_real = self.marker_tiles[key]
                dist = self.calculate_dist(warped_marker_detected, marker_real)
                sim = 1/dist
                similarities[i][key] = sim
        #print("similarities", similarities)
        return np.array(similarities)
    
    
    @staticmethod
    def calculate_dist(marker_detected, marker_real):
        """
        Returns the sum of distances from each detected marker point
        to the closest point in the real marker template.
        """
        marker_detected = np.asarray(marker_detected, dtype=np.float32)
        marker_real = np.asarray(marker_real, dtype=np.float32)

        total_dist = 0.0

        for px, py in marker_detected:
            dists = np.linalg.norm(marker_real - np.array([px, py]), axis=1)
            total_dist += np.min(dists)

        return total_dist
