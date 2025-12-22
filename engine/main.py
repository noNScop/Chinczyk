import os
import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from helpers import input_videos, find_main_folder

from detectors.pawn_detector import PawnDetector
from detectors.playerTurn_die_detector import PlayerTurnDieDetector
from detectors.die_handler import Die_handler
from detectors.board_detector import BoardDetector
from detectors.internal_board_detector import InternalBoardDetector
from detectors.turn_detector import TurnDetector
from overlays.corner_overlay import CornerOverlay
from overlays.video_overlay import VideoOverlay


class GameState:
    # Each object will likely need some internal memory to account for situation when it disappears from the view,
    # so instead of python primitives like strings we will need a class for each object
    def __init__(self):
        self.turn = None # green / blue

        # Each can be -  # of tile / # home tile / base
        self.green_pawns = [None, None, None, None]
        self.blue_pawns = [None, None, None, None]

        self.die = None # most recent number on the die

        self.throws = None # Number of throws remaining for the current player 0-3

    def update_from_frame(self, frame):
        # For now it is a frame, but we may need to analyze multiple frames
        pass





class EventDetector:
    # Likely each event will require separatew detector like object detectors
    # Track events like pawn anihilation, or exiting the base
    # Separate class because I believe that events can be treated independently to the game state
    def __init__(self):
        pass



def main():
    SKIP_FRAMES = 10 # only in 1/SKIP_FRAMES the most heavy computationaly thisgs will be performed
    videos = input_videos()
    main_folder = find_main_folder()

    print("Videos to be processed:", *videos, sep="\n")

    os.makedirs(os.path.join(main_folder, "output"), exist_ok=True)

    for video in videos:
        video_name = Path(video).name
        cap = cv2.VideoCapture(video)
        die_handler = Die_handler() 
        board_detector = BoardDetector()
        internal_board = InternalBoardDetector()
        turn_detector = TurnDetector()


        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_writer = cv2.VideoWriter(os.path.join(main_folder, 'output', video_name), fourcc, fps, (width, height))
        if not output_writer.isOpened():
            print("VideoWriter failed to open:", f"output/{video_name}")
            continue

        for frame_i in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            output_frame = frame.copy()
            if not ret:
                break

            pawn_centers_green = PawnDetector.find_green_pawns(frame)
            pawn_centers_blue = PawnDetector.find_blue_pawns(frame)

            board_detector.update(frame)
            if board_detector.ready:
                M = board_detector.get_M()
                M_inv = board_detector.get_M_inv()
                if frame_i % SKIP_FRAMES == 1:
                    internal_board.update_occupied_dicts(M, pawn_centers_green, pawn_centers_blue)
                    internal_board.update_unwarped_overlay(frame, M_inv)
            board_corners = board_detector.board_corners
            

            pts_arr, labels = PlayerTurnDieDetector.find_objects(frame)
            for i in range(len(labels)):
                label, pts = labels[i], pts_arr[i]

                if label == 'die':
                    die_handler.update(frame, pts)
                    output_frame = VideoOverlay.draw_die(output_frame, pts, die_handler)
                
                if label == 'marker':
                    turn_detector.add_marker(pts)
                    output_frame = VideoOverlay.draw_marker(output_frame, pts)
                
                if label == 'reflection':
                    output_frame = VideoOverlay.draw_reflection(output_frame, pts)
            
            if board_detector.ready:
                M = board_detector.get_M()
                turn_detector.decide_on_turn(M)

                
                



            # Draw detected pawns
            output_frame = VideoOverlay.draw_green_pawn_circles(output_frame, pawn_centers_green)
            output_frame = VideoOverlay.draw_blue_pawn_circles(output_frame, pawn_centers_blue)

            output_frame = VideoOverlay.draw_board_boarder(output_frame, ordered=board_corners)

            output_frame = VideoOverlay.draw_tile_hulls(
                frame=output_frame,
                internal_board=internal_board,
                draw_alpha=0.5
            )

            output_frame = CornerOverlay.draw_turn_info(output_frame, turn_detector,)
            
            output_writer.write(output_frame)

        cap.release()
        output_writer.release()
        print(f"Finished writing video: output/{video_name}")

if __name__ == "__main__":
    main()