import os
import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial

from helpers import input_videos, find_main_folder

from detectors.pawn_detector import PawnDetector
from detectors.playerTurn_die_detector import PlayerTurnDieDetector
from detectors.die_handler import Die_handler
from detectors.board_detector import BoardDetector
from detectors.internal_board_detector import InternalBoardDetector

from state_controllers.turn_state_controller import TurnStateController
from state_controllers.pawn_state_controller import PawnStateController

from event_recognizers.win_game_recognizer import WinGameRecognizer
from event_recognizers.die_throw_recognizer import DieThrowRecognizer
from event_recognizers.enter_home_recognizer import EnterHomeRecognizer
from event_recognizers.leave_base_recognizer import LeaveBaseRecognizer

from overlays.corner_overlay import CornerOverlay
from overlays.video_overlay import VideoOverlay
from overlays.event_overlay import EventOverlay

# aim flow:
# [Detectors]  →  [StateControllers]  →  [GameState]  →  [Overlay / Logic]

class GameState:
    # Each object will likely need some internal memory to account for situation when it disappears from the view,
    # so instead of python primitives like strings we will need a class for each object

    def __init__(self, internal_board: InternalBoardDetector):
        self.turn = None # green / blue

        # Each can be -  # of tile / # home tile / base
        #self.green_pawns = [None, None, None, None] 
        #self.blue_pawns = [None, None, None, None]
        # as we dont differentiate pawns of player i am not sure if the above would work
        self.blue_home = None # num of blue pawns in home
        self.green_home = None
        self.blue_base = None
        self.green_base = None


        self.die = None # most recent number on the die

        self.throws = None # Number of throws remaining for the current player 0-3

        self.internal_board_detector

    def update(self, turn, die_score, ):
        # maybe allow other classes accumulate info from may frames and decide on true state, here we will jusst gather info
        self.turn = self.update_turn(turn)
        self.die = self.update_die(die_score)



    def update_turn(self, turn):
        self.turn = turn

    def update_die(self, die_score):
        self.die = die_score


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
    tiles = np.load(os.path.join(main_folder,"engine/board_data/regularTiles_gameModel.npy"), allow_pickle=True).item()
    tiles_green = np.load(os.path.join(main_folder,"engine/board_data/greenHomeTiles_gameModel.npy"), allow_pickle=True).item()
    tiles_blue = np.load(os.path.join(main_folder,"engine/board_data/blueHomeTiles_gameModel.npy"), allow_pickle=True).item()
    marker_tiles = np.load(os.path.join(main_folder,"engine/board_data/markerTiles_gameModel.npy"), allow_pickle=True).item() 
    board_bgr = cv2.imread(os.path.join(main_folder, "data", "board.jpg"))
    board_relaxed_bgr = cv2.imread(os.path.join(main_folder, "data", "board_relaxed.jpg"))



    print("Videos to be processed:", *videos, sep="\n")

    os.makedirs(os.path.join(main_folder, "output"), exist_ok=True)

    for video in videos:
        video_name = Path(video).name
        cap = cv2.VideoCapture(video)

        die_handler = Die_handler() 
        board_detector = BoardDetector()
        internal_board = InternalBoardDetector(tiles, tiles_blue, tiles_green, board_relaxed_bgr)

        turn_state = TurnStateController(marker_tiles)
        pawn_state = PawnStateController(tiles, tiles_blue, tiles_green, turn_state)

        die_throw_recognizer = DieThrowRecognizer()
        win_recognizer = WinGameRecognizer(pawn_state)
        enter_home_recognizer = EnterHomeRecognizer(pawn_state)
        leave_base_recognizer = LeaveBaseRecognizer(pawn_state)

        event_overlay = EventOverlay()



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

            pawn_centers_green_stable = pawn_centers_green #placeholder, maybe there should be class to make positions consistent
            pawn_centers_blue_stable = pawn_centers_blue




            board_detector.update(frame)
            if board_detector.ready:
                M = board_detector.get_M()
                M_inv = board_detector.get_M_inv()
                internal_board.update_occupied_dicts(M, pawn_centers_green_stable, pawn_centers_blue_stable)

                if frame_i % SKIP_FRAMES == 1:
                    internal_board.update_unwarped_overlay(frame, M_inv)
            board_corners = board_detector.board_corners
            

            dice_all = [] # probably should fix so only 1 die can be detected
            marker_all = []
            reflection_all = []
            if_die_detected_this_frame = False

            pts_arr, labels = PlayerTurnDieDetector.find_objects(frame)
            for i in range(len(labels)):
                label, pts = labels[i], pts_arr[i]

                if label == 'die':
                    die_handler.update(frame, pts)
                    dice_all.append(pts)
                    if_die_detected_this_frame = True # chyba trochę za dużo logiki tu się dzieje, coś z tym zrobić?
                
                if label == 'marker':
                    turn_state.add_marker(pts)
                    marker_all.append(pts)
                
                if label == 'reflection':
                    reflection_all.append(pts)
            if board_detector.ready:
                die_throw_recognizer.update(die_handler.get_number(), if_die_detected_this_frame, frame_i, M, dice_all, board_corners)

            if board_detector.ready:
                turn_state.decide_on_turn(M)



            if board_detector.ready:
                pawn_state.update(pawn_centers_green_stable, pawn_centers_blue_stable, internal_board.occupied_tiles)

                
                



            # Draw detected pawns
            output_frame = VideoOverlay.draw_green_pawn_circles(output_frame, pawn_centers_green_stable)
            output_frame = VideoOverlay.draw_blue_pawn_circles(output_frame, pawn_centers_blue_stable)

            for pts in dice_all:
                output_frame = VideoOverlay.draw_die(output_frame, pts, die_handler)
            for pts in marker_all:
                output_frame = VideoOverlay.draw_marker(output_frame, pts)
            for pts in reflection_all:
                output_frame = VideoOverlay.draw_reflection(output_frame, pts)

            if board_detector.ready:
                output_frame = VideoOverlay.draw_board_boarder(output_frame, ordered=board_corners)
                output_frame = VideoOverlay.draw_tile_hulls(frame=output_frame, internal_board=internal_board, draw_alpha=0.5)



            output_frame = CornerOverlay.draw_turn_info(output_frame, turn_state,)
            output_frame = CornerOverlay.draw_pawn_info(output_frame, pawn_state,)
            
            #####################################
            # Events
            #####################################

            print("event ", die_throw_recognizer.which_event(), die_throw_recognizer.frame_num - die_throw_recognizer.last_event_frame)

            if die_throw_recognizer.if_event:
                event_overlay.add_event(
                "throwed die:" + str(die_throw_recognizer.which_event()) ,
                effect_func=EventOverlay.outline_text,
                duration=50,)

            if win_recognizer.update():
                winner = win_recognizer.get_winner()
                event_overlay.add_event(
                    f"{winner.upper()} WINS!",
                    effect_func=EventOverlay.slide_up, 
                    duration=100 )
            if turn_state.turn is not None:
                if enter_home_recognizer.update(TurnStateController.ID_MARKER_MAPPING[turn_state.turn]):
                    event_overlay.add_event(
                        f"{enter_home_recognizer.last_entered} pawn entered the home!",
                        effect_func=partial(EventOverlay.fade_center, font_scale=2.5),
                        duration=100
                    )

            if leave_base_recognizer.update():
                event_overlay.add_event(
                    f"{leave_base_recognizer.player} pawn left the base!",
                    effect_func=partial(EventOverlay.fade_center, font_scale=2.5)
                )

            # if pawn_capture_recognizer.update():
            #     event_overlay.add_event(
            #         f"{pawn_capture_recognizer.prey} pawn has been slayed!",
            #         effect_func=partial(EventOverlay.bounce_text, font_scale=2.5)
            #     )




                        
            output_frame = event_overlay.draw(output_frame) 
            output_writer.write(output_frame)

        cap.release()
        output_writer.release()
        print(f"Finished writing video: output/{video_name}")

if __name__ == "__main__":
    main()