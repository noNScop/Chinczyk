import cv2
import numpy as np
import os
from helpers import find_main_folder

class InternalBoardDetector():
    def __init__(self, tiles, tiles_blue, tiles_green, board_relaxed_bgr, canonical_size = 800):
        self.tiles = tiles
        self.tiles_blue = tiles_blue
        self.tiles_green = tiles_green
        self.board_relaxed_bgr = board_relaxed_bgr
        self.canonical_size = canonical_size

        all_tiles = {}
        all_tiles.update(tiles)        
        all_tiles.update(tiles_blue)  
        all_tiles.update(tiles_green)  
        self.all_tiles = all_tiles

        self.regions= self.initiate_regions()
        self.regions_dict = None
        self.tile_regions_dict = self.regions_with_tile_points(self.regions, self.all_tiles) 
        # tile_regions_dict is dict tile_id: region_id (which regions belong to which tiles)
        self.occupied_tiles = {}
        self.last_unwarped_overlay = None


    def update_occupied_dicts(self, M, green_pawns_orig, blue_pawns_orig):
        self.occupied_tiles = self.detect_occupied_tiles_separate(green_pawns_orig, blue_pawns_orig, M)
        # occupied example {47: 'green', 16: 'green', 19: 'green', 38: 'blue', 42: 'blue'}


    def initiate_regions(self):
        """
        Detect regions from a grayscale image using Canny + morphological closure + flood fill.
        Returns:
            regions: 2D int32 array of region labels
            num_regions: total number of regions (next unused label)
        """

        board_relaxed_bgr_canonical = cv2.resize(self.board_relaxed_bgr, (self.canonical_size, self.canonical_size))
        board_relaxed_rgb = cv2.cvtColor(board_relaxed_bgr_canonical, cv2.COLOR_BGR2RGB)

        ludo_gray = cv2.cvtColor(board_relaxed_rgb, cv2.COLOR_RGB2GRAY)

        # Compute Canny thresholds
        high, _ = cv2.threshold(ludo_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low = 0.5 * high

        edges = cv2.Canny(ludo_gray, low, high, apertureSize=3)
        # Stronger morphology to close edges
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))

        # Flood-fill regions
        regions = np.zeros(edges.shape[:2], np.int32)
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def find_neighbours(y, x):
            c_neighbours = []
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= edges.shape[0] or nx < 0 or nx >= edges.shape[1]:
                    continue
                if regions[ny, nx] > 0:
                    continue
                if edges[ny, nx] == 255:  # boundary
                    continue
                c_neighbours.append((ny, nx))
            return c_neighbours

        def grow_region(y, x, cls):
            stack = [(y, x)]
            regions[y, x] = cls
            while stack:
                cy, cx = stack.pop()
                for ny, nx in find_neighbours(cy, cx):
                    regions[ny, nx] = cls
                    stack.append((ny, nx))

        cls = 1
        for y in range(edges.shape[0]):
            for x in range(edges.shape[1]):
                if regions[y, x] == 0 and edges[y, x] == 0:
                    grow_region(y, x, cls)
                    cls += 1

        return regions
    

    @staticmethod
    def regions_with_tile_points(regions, tiles_dict):
        """
        regions: 2D array of labels
        tiles_dict: dict {tile_id: [[x,y], ...]}
        Returns a dict: tile_id -> set of region labels that contain points of the tile
        """
        tile_regions = {}
        for tile_id, pts in tiles_dict.items():
            tile_regions[tile_id] = set()
            for x, y in pts:
                y_px, x_px = int(y), int(x)
                if 0 <= y_px < regions.shape[0] and 0 <= x_px < regions.shape[1]:
                    tile_regions[tile_id].add(regions[y_px, x_px])
        return tile_regions


    def detect_occupied_tiles_separate(self, pawn_coords_green, pawn_coords_blue, M,  distance_limit=60):
        """
        Detect which tiles are occupied by green and blue pawns separately.
        
        Returns:
            dict with tile_id -> BGR color
        """        
        occupied_tiles = {}

        def assign_tiles(pawn_coords, color):
            if len(pawn_coords) == 0:
                return
            pts = np.array(pawn_coords, dtype=np.float32).reshape(-1, 1, 2)
            warped_pts = cv2.perspectiveTransform(pts, M).reshape(-1, 2) # heavy computationally

            for px, py in warped_pts:
                px, py = int(px), int(py)
                closest_tile = None
                closest_dist = float("inf")
                for tile_id, points in self.all_tiles.items():
                    for tx, ty in points:
                        d = np.linalg.norm(np.array([px, py]) - np.array([tx, ty]))
                        if d < closest_dist:
                            closest_dist = d
                            closest_tile = tile_id
                if closest_dist <= distance_limit:
                    occupied_tiles[closest_tile] = color

        assign_tiles(pawn_coords_green, 'green')  # green
        assign_tiles(pawn_coords_blue, 'blue')   # blue

        return occupied_tiles


    def update_unwarped_overlay(self, frame, M_inv, fill_alpha=0.3, outline_alpha=1.0, thickness=5):
        """
        Draw convex hulls of tiles with fill and outline alpha, then warp overlay back to original frame.
        
        Parameters:
        - frame: original frame (H, W, 3)
        - M_inv: inverse perspective matrix (warped -> original)
        - fill_alpha: transparency of filled tile color
        - outline_alpha: transparency of hull outline
        - thickness: outline thickness
        """
        # Use uint8 for OpenCV compatibility
        overlay = np.zeros((self.canonical_size, self.canonical_size, 3), dtype=np.uint8)

        for tile_id, region_ids in self.tile_regions_dict.items():
            all_points = []

            # collect all points of all regions for this tile
            for r in region_ids:
                ys, xs = np.where(self.regions == r)
                all_points.extend(list(zip(xs, ys)))

            if len(all_points) < 3:
                continue  # need at least 3 points for convex hull

            all_points = np.array(all_points, dtype=np.int32)
            hull = cv2.convexHull(all_points)

            # Determine tile color
            if tile_id in self.occupied_tiles:
                if self.occupied_tiles[tile_id] == 'green':
                    color = np.array([0, 255, 0], dtype=np.uint8)
                elif self.occupied_tiles[tile_id] == 'blue':
                    color = np.array([0, 0, 255], dtype=np.uint8)
                else:
                    color = np.array([200, 200, 200], dtype=np.uint8)
            else:
                color = np.array([200, 200, 200], dtype=np.uint8)

            # Fill tile with alpha
            mask = np.zeros((self.canonical_size, self.canonical_size), dtype=np.uint8)
            cv2.fillPoly(mask, [hull], 1)
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = ((fill_alpha * color + (1 - fill_alpha) * overlay[mask_bool])).astype(np.uint8)

            # Draw outline with alpha
            outline_img = np.zeros_like(overlay)
            cv2.polylines(outline_img, [hull], isClosed=True, color=color.tolist(), thickness=thickness)
            overlay = np.where(outline_img > 0,
                            (outline_alpha * outline_img + (1 - outline_alpha) * overlay).astype(np.uint8),
                            overlay)

        # Warp back to original frame
        self.last_unwarped_overlay = cv2.warpPerspective(
            overlay,
            M_inv,
            (frame.shape[1], frame.shape[0])
        )

