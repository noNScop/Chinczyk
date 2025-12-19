def input_videos():
    all_videos = [
        "./data/vid_1_dist=1.mp4",
        "./data/vid_2_dist=1.mp4",
        "./data/vid_3_dist=1.mp4",
        "./data/vid_1_dist=2.mp4",
        "./data/vid_2_dist=2.mp4",
        "./data/vid_3_dist=2.mp4",
        "./data/vid_1_dist=3.mp4",
        "./data/vid_2_dist=3.mp4",
        "./data/vid_3_dist=3.mp4",
    ]

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
                return all_videos
            if n in observed:
                continue
            
            if n < 0 or n > 9:
                print("Error: numbers must be in range 0-9.")
                videos = []
                break
            
            videos.append(all_videos[n])
            observed.add(n)

        if not videos:
            continue

        return videos
