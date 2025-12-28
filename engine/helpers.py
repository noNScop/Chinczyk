import os
from pathlib import Path

def find_main_folder():
    """
    Locate the project root directory.

    This helper exists to make the codebase independent of the current working
    directory. It allows the project to be run from different entry points
    (e.g. IDE, terminal, or script execution) while still correctly resolving
    paths to the `data/` directory.

    Search strategy:
    1. Check if `data/` exists in the current working directory.
    2. Check if `data/` exists relative to this file's location.
    3. Raise an explicit error if neither location is valid.

    Returns:
        Path to the project root directory (parent of `data/`).

    Raises:
        FileNotFoundError: If the `data/` directory cannot be located.
    """

    # Case 1: project executed from the repository root
    cwd_data = Path.cwd() / "data"
    if cwd_data.is_dir():
        return cwd_data.parent

    # Case 2: project executed from a submodule or script
    file_data = Path(__file__).resolve().parent.parent / "data"
    if file_data.is_dir():
        return file_data.parent

    # Neither location contains the expected directory structure
    raise FileNotFoundError(
        "Could not find 'data' folder in CWD or script parent directory"
    )


def input_videos():
    """
    Interactive helper for selecting input videos.

    The function presents a fixed list of known video filenames and allows
    the user to select:
      - specific videos by index (1–9), or
      - all videos at once using `0`.

    The function performs basic validation and prevents duplicate selections.

    Returns:
        List[str]: Absolute paths to the selected video files.
    """

    # Ordered list of available video filenames
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

    # Resolve absolute paths to all videos
    all_videos_paths = [os.path.join(find_main_folder(), 'data', v) for v in all_videos]

    while True:
        raw = input("Choose videos from 1 to 9 (space-separated) or 0 - all: ")
        
        # Parse user input
        try:
            numbers = [int(x) for x in raw.split()]
        except ValueError:
            print("Error: all inputs must be integers.")
            continue

        # Optional validation (kept explicit and readable)
        observed = set()
        videos = []
        for n in numbers:
            # Select all videos
            if n == 0:
                return all_videos_paths
            
            # Ignore duplicate indices
            if n in observed:
                continue
            
            # Validate index range
            if n < 0 or n > 9:
                print("Error: numbers must be in range 0-9.")
                videos = []
                break
            
            videos.append(all_videos_paths[n -1])
            observed.add(n)

        # Retry if input was invalid or empty
        if not videos:
            continue

        return videos
