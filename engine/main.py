import os
import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from helpers import input_videos, find_main_folder

from detectors.pawn_detector import PawnDetector
from detectors.playerTurn_die_detector import PlayerTurnDieDetector
from detectors.die_handler import Die_handler


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
    
    @staticmethod
    def draw_die(frame, pts, die_handler : Die_handler):

        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        num = die_handler.get_number()

        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(
            frame,
            f"{int(num)}",
            (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        circles = die_handler.get_circes()
        if circles is not None:
            for x, y, r in circles:
                x, y, r = int(x), int(y), int(r)
                cv2.circle(frame, (x, y), r, (0, 0, 255), 2)   # circle
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1) # center
                cv2.putText(
                    frame,
                    f"r={r}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1
                )
        return frame
    
    @staticmethod
    def draw_reflection(frame, pts):
        cv2.polylines(frame, [pts], True, (255, 255, 0), 2)  # cyan outline
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(
            frame,
            "reflection",
            (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        return frame

    @staticmethod
    def draw_marker(frame, pts):
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)  # red outline
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(
            frame,
            "marker",
            (cx - 20, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        return frame

def main():
    videos = input_videos()
    main_folder = find_main_folder()

    print("Videos to be processed:", *videos, sep="\n")

    os.makedirs(os.path.join(main_folder, "output"), exist_ok=True)

    for video in videos:
        video_name = Path(video).name
        cap = cv2.VideoCapture(video)
        die_handler = Die_handler() 

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
            if not ret:
                break

            pawn_centers_green = PawnDetector.find_green_pawns(frame)
            pawn_centers_blue = PawnDetector.find_blue_pawns(frame)


            draw_map = {
                'die': lambda f, p: VideoOverlay.draw_die(f, p, die_handler),
                'marker': lambda f, p: VideoOverlay.draw_marker(f, p),
                'reflection': lambda f, p: VideoOverlay.draw_reflection(f, p),
            }

            pts_arr, labels = PlayerTurnDieDetector.find_objects(frame)
            for i in range(len(labels)):
                label, pts = labels[i], pts_arr[i]
                if label == 'die':
                    die_handler.update(frame, pts)
                func = draw_map.get(label)
                if func:
                    func(frame, pts)



            # Draw detected pawns
            output_frame = VideoOverlay.draw_green_pawn_circles(frame, pawn_centers_green)
            output_frame = VideoOverlay.draw_blue_pawn_circles(output_frame, pawn_centers_blue)


            output_writer.write(output_frame)

        cap.release()
        output_writer.release()
        print(f"Finished writing video: output/{video_name}")

if __name__ == "__main__":
    main()