import cv2
import numpy as np
from collections import deque

from game_state.move_suggester import MoveSuggester

class TurnStateController():
    """
    Class responsible for tracking and determining the current player's turn
    based on visual detection of markers on the board.  
    It maintains a short history of similarity scores to make robust decisions.
    """

    # Mapping marker IDs to player colors
    ID_MARKER_MAPPING = {0: "blue", 1:"green"}

    def __init__(self, marker_tiles, move_suggester: MoveSuggester):
        """
        Parameters
        ----------
        marker_tiles : dict
            Dictionary containing real marker positions keyed by player ID (0=blue, 1=green).
        move_suggester : MoveSuggester
            Reference to MoveSuggester to stop suggestions when turn changes.
        """

        self.marker_tiles = marker_tiles
        self.markers_all = [] # Detected markers in current frame (may be >1 due to detection noise)
        self.turn = None  # Currently decided turn: 0=blue, 1=green

        # History of similarities for robust decision-making, each entry = [sim_blue, sim_green]
        self.sim_history = deque(maxlen=30)
        
        # Reference to move suggester to stop suggestions when turn changes
        self.move_suggerster = move_suggester


    def reset_internal_markers(self):
        self.markers_all = []


    def add_marker(self, detected_marker):
        self.markers_all.append(detected_marker)


    @staticmethod
    def warp_points(points, M):
        """
        Applies a homography matrix to warp points to board coordinates.

        Parameters
        ----------
        points : np.ndarray
            Points to warp.
        M : np.ndarray
            3x3 homography matrix.

        Returns
        -------
        np.ndarray
            Warped points in board coordinate system.
        """

        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        return warped_pts


    def decide_on_turn(self, M):
        """
        Determine the current turn based on detected markers and similarity to templates.
        Updates self.turn and stops move suggestions if turn changed.

        Parameters
        ----------
        M : np.ndarray
            Homography matrix to warp detected markers to board coordinates.
        """

        if(len(self.markers_all)) == 0:
            return # No markers detected, cannot decide
        
        # Compute similarity scores for each detected marker
        similarities = self.calculate_similarities(M)
        sim_blue = np.max(similarities[:, 0])
        sim_green = np.max(similarities[:, 1])
        self.sim_history.append([sim_blue, sim_green])

        # Reset markers for next frame
        self.reset_internal_markers()

        # Decide turn based on history
        turn = self._decide_from_memory()
        if turn != self.turn:
            self.move_suggerster.stop_suggestion() # Stop previous turn suggestions if turn changed
        self.turn = turn


    def _decide_from_memory(self):
        """
        Decide the turn based on accumulated similarity history.

        Returns
        -------
        int
            Turn ID: 0 = blue, 1 = green
        """

        sims = np.array(self.sim_history)

        total_blue = sims[:, 0].sum()
        total_green = sims[:, 1].sum()

        return 0 if total_blue > total_green else 1
    

    def calculate_similarities(self, M):
        """
        Calculate similarity between detected markers and real marker templates.

        Parameters
        ----------
        M : np.ndarray
            Homography matrix to warp detected markers.

        Returns
        -------
        np.ndarray
            Array of shape (num_markers, 2) containing similarity scores [blue, green]
        """

        similarities = [[0 for _ in range(2)] for _ in range(len(self.markers_all)) ]
        for i, marker_detected in enumerate(self.markers_all):
            warped_marker_detected = self.warp_points(marker_detected, M)
            for key in self.marker_tiles:
                marker_real = self.marker_tiles[key]
                dist = self.calculate_dist(warped_marker_detected, marker_real)
                sim = 1 / dist # Higher similarity for smaller distance
                similarities[i][key] = sim

        return np.array(similarities)
    
    
    @staticmethod
    def calculate_dist(marker_detected, marker_real):
        """
        Calculate the sum of distances from each detected marker point
        to the closest point in the real marker template.

        Parameters
        ----------
        marker_detected : np.ndarray
            Warped detected marker points.
        marker_real : np.ndarray
            Points of the template marker.

        Returns
        -------
        float
            Total distance score (smaller = more similar)
        """
        
        marker_detected = np.asarray(marker_detected, dtype=np.float32)
        marker_real = np.asarray(marker_real, dtype=np.float32)

        total_dist = 0.0

        for px, py in marker_detected:
            dists = np.linalg.norm(marker_real - np.array([px, py]), axis=1)
            total_dist += np.min(dists)

        return total_dist
