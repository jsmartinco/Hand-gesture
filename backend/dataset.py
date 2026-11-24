"""Loads the extracted landmark files (data/landmarks/*.json) as training data.

The training script imports load_dataset() from here. Every recording goes
through preprocess.py, the same step the API uses for webcam recordings.

Rule: a video is used only if MediaPipe found a hand in at least
MIN_HAND_FRAMES frames (30 frames = 1 second at 30 fps). Videos below that
lost the hand for most of the sign, so they would teach the model mostly zeros.
They are skipped and listed, never deleted.

Quick check (from the backend folder, virtual environment active):
    python dataset.py
"""

import json
from pathlib import Path

import numpy as np

from preprocess import preprocess

HERE = Path(__file__).parent
DEFAULT_LANDMARKS = HERE / "data" / "landmarks"

MIN_HAND_FRAMES = 30


def hand_frame_count(recording):
    return sum(1 for f in recording["frames"] if f.get("left") or f.get("right"))


def load_dataset(folder=DEFAULT_LANDMARKS, min_hand_frames=MIN_HAND_FRAMES):
    """Returns (X, y, labels, ids, skipped).

    X       float32 array (videos, 60, 126), the preprocessed recordings
    y       int array (videos,), index into labels
    labels  sorted list of sign names, so y = 3 means labels[3]
    ids     video id of each row (to keep all of one video in the same split)
    skipped list of (video id, sign, hand frames) left out by the rule
    """
    features, signs, ids, skipped = [], [], [], []
    for path in sorted(Path(folder).glob("*.json")):
        recording = json.loads(path.read_text(encoding="utf-8"))
        hands = hand_frame_count(recording)
        if hands < min_hand_frames:
            skipped.append((recording["videoId"], recording["label"], hands))
            continue
        features.append(preprocess(recording))
        signs.append(recording["label"])
        ids.append(recording["videoId"])

    labels = sorted(set(signs))
    y = np.array([labels.index(s) for s in signs])
    X = np.stack(features).astype(np.float32)
    return X, y, labels, ids, skipped


if __name__ == "__main__":
    X, y, labels, ids, skipped = load_dataset()
    print(f"X {X.shape}, {len(labels)} signs")
    for i, sign in enumerate(labels):
        print(f"  {sign:<12} {np.sum(y == i):>3} videos")
    print(f"\nSkipped (fewer than {MIN_HAND_FRAMES} frames with a hand): {len(skipped)}")
    for video_id, sign, hands in skipped:
        print(f"  {video_id}  {sign:<12} {hands} frames with a hand")
