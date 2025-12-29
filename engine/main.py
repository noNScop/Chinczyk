import os
import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from helpers import input_videos, find_main_folder

from detectors.die_handler import Die_handler
from detectors.pawn_detector import PawnDetector
from detectors.board_detector import BoardDetector
from detectors.playerTurn_die_detector import PlayerTurnDieDetector
from detectors.internal_board_detector import InternalBoardDetector

from state_controllers.turn_state_controller import TurnStateController
from state_controllers.pawn_state_controller import PawnStateController

from event_recognizers.win_game_recognizer import WinGameRecognizer
from event_recognizers.die_throw_recognizer import DieThrowRecognizer
from event_recognizers.enter_home_recognizer import EnterHomeRecognizer
from event_recognizers.leave_base_recognizer import LeaveBaseRecognizer
from event_recognizers.pawn_capture_recognizer import PawnCaptureRecognizer

from overlays.video_overlay import VideoOverlay
from overlays.event_overlay import EventOverlay
from overlays.corner_overlay import CornerOverlay

from game_state.move_suggester import MoveSuggester

def main():
    # ============================================================
    # Configuration
    # ============================================================

    # Perform heavy computations only once every SKIP_FRAMES frames
    # (e.g. expensive warping / overlays that do not need per-frame updates)

    SKIP_FRAMES = 3

    # Input videos to process
    videos = input_videos()

    # Root project directory (used to resolve all data paths)
    main_folder = find_main_folder()

    # ============================================================
    # Static game model data (precomputed, loaded once)
    # ============================================================

    # Board tile mappings used by the internal game model
    tiles = np.load(
        os.path.join(main_folder,"engine/board_data/regularTiles_gameModel.npy"), 
        allow_pickle=True
    ).item()

    tiles_green = np.load(
        os.path.join(main_folder,"engine/board_data/greenHomeTiles_gameModel.npy"), 
        allow_pickle=True
    ).item()

    tiles_blue = np.load(
        os.path.join(main_folder,"engine/board_data/blueHomeTiles_gameModel.npy"), 
        allow_pickle=True
    ).item()

    marker_tiles = np.load(
        os.path.join(main_folder,"engine/board_data/markerTiles_gameModel.npy"), 
        allow_pickle=True
    ).item()

    # Reference board images
    board_relaxed_bgr = cv2.imread(os.path.join(main_folder, "data", "board_relaxed.jpg"))

    print("Videos to be processed:", *videos, sep="\n")

    # Output directory
    os.makedirs(os.path.join(main_folder, "output"), exist_ok=True)

    # ============================================================
    # Main video processing loop
    # ============================================================
    for video in videos:
        video_name = Path(video).name
        cap = cv2.VideoCapture(video)

        # --------------------------------------------------------
        # Core game logic / state managers
        # --------------------------------------------------------
        move_suggester = MoveSuggester()
        die_handler = Die_handler()

        board_detector = BoardDetector()
        internal_board = InternalBoardDetector(tiles, tiles_blue, tiles_green, board_relaxed_bgr, move_suggester)

        # State controllers
        turn_state = TurnStateController(marker_tiles, move_suggester)
        pawn_state = PawnStateController(tiles, tiles_blue, tiles_green, turn_state, board_detector)

        # --------------------------------------------------------
        # Event recognizers (stateless detectors + temporal logic)
        # --------------------------------------------------------
        die_throw_recognizer = DieThrowRecognizer(move_suggester)
        win_recognizer = WinGameRecognizer(pawn_state)
        enter_home_recognizer = EnterHomeRecognizer(pawn_state)
        leave_base_recognizer = LeaveBaseRecognizer(pawn_state)
        pawn_capture_recognizer = PawnCaptureRecognizer(pawn_state)

        # Overlay manager for animated / temporal UI events
        event_overlay = EventOverlay()

        # --------------------------------------------------------
        # Video I/O setup
        # --------------------------------------------------------
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_writer = cv2.VideoWriter(os.path.join(main_folder, 'output', video_name), fourcc, fps, (width, height))

        if not output_writer.isOpened():
            print("VideoWriter failed to open:", f"output/{video_name}")
            continue

        # ========================================================
        # Frame-by-frame processing
        # ========================================================
        for frame_i in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Copy frame for drawing overlays
            output_frame = frame.copy()

            # ----------------------------------------------------
            # Pawn detection (color-based)
            # ----------------------------------------------------
            pawn_centers_green_stable = PawnDetector.find_green_pawns(frame)
            pawn_centers_blue_stable = PawnDetector.find_blue_pawns(frame)

            # ----------------------------------------------------
            # Board detection and homography update
            # ----------------------------------------------------
            board_detector.update(frame)

            if board_detector.ready:
                M = board_detector.get_M()
                M_inv = board_detector.get_M_inv()

                # Update internal board state using detected pawn positions
                internal_board.update_occupied_dicts(M, pawn_centers_green_stable, pawn_centers_blue_stable)

                # Expensive visualizations updated sparsely
                if frame_i % SKIP_FRAMES == 1:
                    internal_board.update_unwarped_overlay(frame, M_inv)

            board_corners = board_detector.board_corners
            
            # ----------------------------------------------------
            # Player marker / die / reflection detection
            # ----------------------------------------------------
            dice_all = []
            marker_all = []
            reflection_all = []
            if_die_detected_this_frame = False

            pts_arr, labels = PlayerTurnDieDetector.find_objects(frame)
            for i in range(len(labels)):
                label, pts = labels[i], pts_arr[i]

                if label == 'die':
                    die_handler.update(frame, pts)
                    dice_all.append(pts)
                    if_die_detected_this_frame = True

                elif label == 'marker':
                    turn_state.add_marker(pts)
                    marker_all.append(pts)

                elif label == 'reflection':
                    reflection_all.append(pts)

            # ----------------------------------------------------
            # Turn logic and dice events
            # ----------------------------------------------------
            if board_detector.ready:
                die_throw_recognizer.update(
                    die_handler.get_number(), 
                    if_die_detected_this_frame, 
                    frame_i, 
                    M, 
                    dice_all, 
                    board_corners
                )

            if board_detector.ready:
                turn_state.decide_on_turn(M)

            # ----------------------------------------------------
            # Pawn state update (logical board positions)
            # ----------------------------------------------------
            if board_detector.ready:
                pawn_state.update(pawn_centers_green_stable, pawn_centers_blue_stable, internal_board.occupied_tiles)

            # ----------------------------------------------------
            # Move suggestions
            # ----------------------------------------------------
            if move_suggester.get_if_suggest():
                move_suggester.make_suggestions(internal_board.occupied_tiles, turn_state.turn, pawn_state.blue_base, pawn_state.green_base )

            # ====================================================
            # Rendering overlays
            # ====================================================
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

            # Corner UI overlays
            output_frame = CornerOverlay.draw_turn_info(output_frame, turn_state,)
            output_frame = CornerOverlay.draw_pawn_info(output_frame, pawn_state,)
            
            # ====================================================
            # Event recognition and animated overlays
            # ====================================================
            if die_throw_recognizer.if_event:
                event_overlay.add_event(
                    f"THROW: {die_throw_recognizer.which_event()}",
                    effect_func=[
                        EventOverlay.flash_bg,
                        EventOverlay.pop_scale,
                        EventOverlay.glow_text
                    ],
                    duration=40
                )

            if win_recognizer.update():
                winner = win_recognizer.get_winner()
                event_overlay.add_event(
                    f"{winner.upper()} WINS!",
                    effect_func=EventOverlay.casino_win,
                    duration=120
                )

            if turn_state.turn is not None and win_recognizer.get_winner() is None and enter_home_recognizer.update(
                    TurnStateController.ID_MARKER_MAPPING[turn_state.turn]):
                event_overlay.add_event(
                    f"{enter_home_recognizer.last_entered} pawn entered HOME!",
                    effect_func=EventOverlay.big_popup,
                    duration=80
                )

            if leave_base_recognizer.update():
                event_overlay.add_event(
                    f"{leave_base_recognizer.player} pawn left the BASE!",
                    effect_func=[
                        EventOverlay.pop_scale,
                        EventOverlay.shake
                    ],
                    duration=60
                )

            if turn_state.turn is not None and pawn_capture_recognizer.update():
                event_overlay.add_event(
                    f"{pawn_capture_recognizer.prey} PAWN SLAYED!",
                    effect_func=[
                        EventOverlay.flash_bg,
                        EventOverlay.shake,
                        EventOverlay.glow_text
                    ],
                    duration=70
                )

            # Draw all active animated events
            output_frame = event_overlay.draw(output_frame)

            # Write frame to output video
            output_writer.write(output_frame)

        cap.release()
        output_writer.release()
        print(f"Finished writing video: output/{video_name}")

if __name__ == "__main__":
    main()