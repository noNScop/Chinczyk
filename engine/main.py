import os
import cv2
from pathlib import Path
from tqdm.auto import tqdm

from helpers import input_videos, find_main_folder

from detectors.pawn_detector import PawnDetector

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

class VideoOverlay:
    # a class with all that is needed to create a visual overlay on the video with the game state and event notifiers (if get's too large m,aybe better to create separate functions in some file / files)
    @staticmethod
    def draw_green_pawn_circles(frame, pawn_centers):
        frame = frame.copy()
        for cx, cy in pawn_centers:
            cv2.circle(frame, (int(cx), int(cy)), 15, (255, 255, 51), 2)

        return frame

    @staticmethod
    def draw_blue_pawn_circles(frame, pawn_centers):
        frame = frame.copy()
        for cx, cy in pawn_centers:
            cv2.circle(frame, (int(cx), int(cy)), 15, (0, 0, 255), 2)

        return frame

def main():
    videos = input_videos()
    main_folder = find_main_folder()

    print("Videos to be processed:", *videos, sep="\n")

    os.makedirs(os.path.join(main_folder, "output"), exist_ok=True)

    for video in videos:
        video_name = Path(video).name
        cap = cv2.VideoCapture(video)

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

        for _ in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            pawn_centers_green = PawnDetector.find_green_pawns(frame)
            pawn_centers_blue = PawnDetector.find_blue_pawns(frame)

            # Draw detected pawns
            output_frame = VideoOverlay.draw_green_pawn_circles(frame, pawn_centers_green)
            output_frame = VideoOverlay.draw_blue_pawn_circles(output_frame, pawn_centers_blue)

            output_writer.write(output_frame)

        cap.release()
        output_writer.release()
        print(f"Finished writing video: output/{video_name}")

if __name__ == "__main__":
    main()