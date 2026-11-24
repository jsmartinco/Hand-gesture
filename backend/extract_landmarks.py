"""Runs MediaPipe over the training videos and saves their hand landmarks.

Each video becomes one JSON file in the SAME format the front end sends
(frames of {t, left, right} plus imageWidth/imageHeight), so preprocess.py
handles training videos and webcam recordings in exactly the same way.

Only raw landmarks are saved here; normalising and resampling happen later
in preprocess.py, so they can be changed without re-running this slow step.

By default it uses the current training stage: the videos copied into
rgb/stage1/<SIGN>/ and their labels in rgb/stage1/labels.json.

Examples (run from the backend folder with the virtual environment active):
    python extract_landmarks.py                              # every sign in the stage
    python extract_landmarks.py --limit 2                    # quick test: 2 videos per sign
    python extract_landmarks.py --signs "HELLO,HELP"         # only some signs
    python extract_landmarks.py --count                      # videos on disk per sign, opens nothing
    python extract_landmarks.py --videos ../rgb --labels ../../Train.json --signs-file signs.txt   # full dataset

Videos already extracted are skipped, so you can stop it and run it again.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
STAGE = HERE.parent / "rgb" / "stage1"
DEFAULT_VIDEOS = STAGE
DEFAULT_LABELS = STAGE / "labels.json"
DEFAULT_OUT = HERE / "data" / "landmarks"
# The same model file the front end uses, so both produce comparable landmarks.
DEFAULT_MODEL = HERE.parent / "hand-gesture-app" / "public" / "models" / "hand_landmarker.task"


def find_videos(videos_dir):
    """Maps video id -> file path for every <id>_kf_rgb.mp4 in videos_dir or its subfolders."""
    return {p.name.removesuffix("_kf_rgb.mp4"): p for p in videos_dir.rglob("*_kf_rgb.mp4")}


def make_landmarker(model_path):
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        # Passing the bytes avoids MediaPipe path problems on Windows.
        base_options=mp_tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def frame_from_result(result, t):
    """Same as frameFromResult in the front end's src/lib/recording.js."""
    frame = {"t": t, "left": None, "right": None}
    for i, hand in enumerate(result.hand_landmarks):
        points = [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in hand]
        side = "left" if result.handedness[i][0].category_name == "Left" else "right"
        # MediaPipe occasionally labels both hands the same; keep both anyway.
        if frame[side] is not None:
            side = "right" if side == "left" else "left"
        frame[side] = points
    return frame


def extract_video(path, model_path):
    """Returns the recording dict for one video."""
    # Imported here so --count works quickly without loading MediaPipe.
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    # A fresh landmarker per video: VIDEO mode tracks hands between frames,
    # and timestamps have to start again for every video.
    with make_landmarker(model_path) as landmarker:
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # OpenCV reads BGR, MediaPipe wants RGB
            t = round(index * 1000 / fps)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
            frames.append(frame_from_result(result, t))
            index += 1
    cap.release()

    return {
        "version": 2,
        "source": "video",
        "imageWidth": width,
        "imageHeight": height,
        "fps": fps,
        "sequenceLength": len(frames),
        "handFrames": sum(1 for f in frames if f["left"] or f["right"]),
        "frames": frames,
    }


def count_videos(labels, videos_dir):
    """Prints how many of each sign's videos are on disk and saves the full list as CSV.

    Only checks whether the files exist; no video is opened.
    """
    videos = find_videos(videos_dir)
    on_disk, listed = Counter(), Counter()
    for video_id, sign in labels.items():
        listed[sign] += 1
        if video_id in videos:
            on_disk[sign] += 1

    rows = sorted(listed, key=lambda s: (-on_disk[s], s))
    csv_file = HERE / "signs_available.csv"
    with csv_file.open("w", encoding="utf-8") as f:
        f.write("sign,videos_on_disk,videos_listed\n")
        for sign in rows:
            f.write(f'"{sign}",{on_disk[sign]},{listed[sign]}\n')

    print("Signs by number of videos on disk:")
    for n in range(max(listed.values()), -1, -1):
        signs_with_n = sum(1 for s in listed if on_disk[s] == n)
        if signs_with_n:
            print(f"  {n:>2} videos: {signs_with_n} signs")
    print(f"\nFull list saved to {csv_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signs", help='comma-separated sign labels as written in Train.json, e.g. "HELLO,THANK YOU"')
    parser.add_argument("--signs-file", type=Path, help="text file with one sign label per line")
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="at most this many videos per sign (for quick tests)")
    parser.add_argument("--count", action="store_true", help="only count the videos on disk per sign and save signs_available.csv")
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))  # {"35776": "WHALE", ...}

    if args.count:
        count_videos(labels, args.videos)
        return

    wanted = set()
    if args.signs:
        wanted |= {s.strip() for s in args.signs.split(",") if s.strip()}
    if args.signs_file:
        wanted |= {line.strip() for line in args.signs_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not wanted:
        wanted = set(labels.values())  # no signs chosen: every sign in the labels file

    unknown = wanted - set(labels.values())
    if unknown:
        parser.error(f"not in {args.labels.name}: {', '.join(sorted(unknown))}")

    videos = find_videos(args.videos)
    args.out.mkdir(parents=True, exist_ok=True)
    found, missing, done, failed = Counter(), Counter(), Counter(), []

    for video_id, sign in labels.items():
        if sign not in wanted:
            continue
        path = videos.get(video_id)
        if path is None:
            missing[sign] += 1
            continue
        if args.limit and found[sign] >= args.limit:
            continue
        found[sign] += 1

        out_file = args.out / f"{video_id}.json"
        if out_file.exists():
            done[sign] += 1
            continue

        try:
            recording = extract_video(path, args.model)
        except Exception as err:  # keep going if one video is broken
            failed.append(f"{path.name}: {err}")
            continue
        recording.update({"videoId": video_id, "label": sign})
        out_file.write_text(json.dumps(recording), encoding="utf-8")
        done[sign] += 1
        print(f"{sign:<20} {path.name}  {recording['handFrames']}/{recording['sequenceLength']} frames with hands")

    print("\nSummary (videos per sign):")
    for sign in sorted(wanted):
        print(f"  {sign:<20} extracted {done[sign]:>3}   not on disk {missing[sign]:>3}")
    if failed:
        print("\nFailed:")
        for line in failed:
            print("  " + line)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
