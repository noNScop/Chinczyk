
from pathlib import Path
import os

def find_main_folder():
    #if data in current working folder
    cwd_data = Path.cwd() / "data"
    if cwd_data.is_dir():
        return cwd_data.parent

    # parent directory of this file
    file_data = Path(__file__).resolve().parent.parent / "data"
    if file_data.is_dir():
        return file_data.parent

    raise FileNotFoundError(
        "Could not find 'data' folder in CWD or script parent directory"
    )


def input_videos():
    all_videos = [
        "vid_1_dist=1.mp4",
        "vid_2_dist=1.mp4",
        "vid_3_dist=1.mp4",
        "vid_1_dist=2.mp4",
        "vid_2_dist=2.mp4",
        "vid_3_dist=2.mp4",
        "vid_1_dist=3.mp4",
        "vid_2_dist=3.mp4",
        "vid_3_dist=3.mp4",
    ]

    all_videos_paths = [os.path.join(find_main_folder(), 'data', v) for v in all_videos]
    while True:
        raw = input("Choose videos from 1 to 9 (space-separated) or 0 - all: ")
        
        try:
            numbers = [int(x) for x in raw.split()]
        except ValueError:
            print("Error: all inputs must be integers.")
            continue

        # Optional validation (kept explicit and readable)
        observed = set()
        videos = []
        for n in numbers:
            if n == 0:
                return all_videos_paths
            if n in observed:
                continue
            
            if n < 0 or n > 9:
                print("Error: numbers must be in range 0-9.")
                videos = []
                break
            
            videos.append(all_videos_paths[n])
            observed.add(n)

        if not videos:
            continue

        return videos
