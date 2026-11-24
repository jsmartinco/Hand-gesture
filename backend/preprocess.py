"""Turns a recording into the fixed-size array the LSTM reads.

Used for BOTH training (videos, via extract_landmarks.py) and prediction
(webcam recordings sent by the front end), so the model always sees data
prepared the same way.

A recording is a dict in the format written by the front end and by
extract_landmarks.py:
    {
      "imageWidth": 640, "imageHeight": 480,
      "frames": [ {"t": 0, "left": [[x, y, z] x 21] or None, "right": ...}, ... ]
    }

Output: numpy array of shape (SEQUENCE_LENGTH, FEATURES_PER_FRAME) = (60, 126):
one row per frame, holding left hand (21 points x 3) then right hand (21 x 3).
A hand that isn't visible is all zeros.
"""

import numpy as np

SEQUENCE_LENGTH = 60
NUM_POINTS = 21
FEATURES_PER_FRAME = 2 * NUM_POINTS * 3

# MediaPipe hand landmark numbers (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
WRIST = 0
MIDDLE_FINGER_BASE = 9


def normalise_hand(points, aspect):
    """Makes one hand independent of where it is in the image and how far it is from the camera.

    1. Aspect ratio: x is a fraction of the image width and y of the height, so on a
       non-square image the hand is stretched. Multiplying x (and z, which MediaPipe
       gives on the same scale as x) by width/height puts everything in the same unit.
    2. Position: subtract the wrist, so the wrist is always at (0, 0, 0).
    3. Distance: divide by the wrist-to-middle-finger-base length, so a hand close
       to the camera and one far away give the same numbers.
    """
    if points is None:
        return np.zeros((NUM_POINTS, 3), dtype=np.float32)

    p = np.array(points, dtype=np.float32)
    if p.shape != (NUM_POINTS, 3):
        raise ValueError(f"Each hand needs {NUM_POINTS} points of [x, y, z], got shape {p.shape}.")

    p[:, 0] *= aspect
    p[:, 2] *= aspect
    p -= p[WRIST]
    hand_size = np.linalg.norm(p[MIDDLE_FINGER_BASE, :2])
    if hand_size > 1e-6:
        p /= hand_size
    return p


def has_hand(frame):
    return frame.get("left") is not None or frame.get("right") is not None


def trim(frames):
    """Drops frames without any hand at the start and end (hands down before/after the sign)."""
    with_hands = [i for i, f in enumerate(frames) if has_hand(f)]
    if not with_hands:
        return []
    return frames[with_hands[0] : with_hands[-1] + 1]


def resample(frames, length=SEQUENCE_LENGTH):
    """Picks `length` frames evenly spread over the recording's duration.

    Uses each frame's timestamp "t" (ms), so a 25-frame clip and a 248-frame clip
    both become `length` frames covering the whole sign. Short clips repeat frames,
    long ones skip frames. Frames without "t" are assumed to be 30 fps.
    """
    times = np.array([f.get("t", i * 1000 / 30) for i, f in enumerate(frames)], dtype=np.float64)
    targets = np.linspace(times[0], times[-1], length)
    # For each target time, the index of the closest real frame.
    nearest = np.abs(times[:, None] - targets[None, :]).argmin(axis=0)
    return [frames[i] for i in nearest]


def preprocess(recording):
    """Recording dict -> array of shape (SEQUENCE_LENGTH, FEATURES_PER_FRAME).

    Raises ValueError with a readable message if the recording can't be used.
    """
    width = recording.get("imageWidth")
    height = recording.get("imageHeight")
    if not width or not height:
        raise ValueError("The recording needs imageWidth and imageHeight.")
    aspect = width / height

    frames = trim(recording.get("frames") or [])
    if not frames:
        raise ValueError("No hand was found in any frame.")

    rows = []
    for frame in resample(frames):
        left = normalise_hand(frame.get("left"), aspect)
        right = normalise_hand(frame.get("right"), aspect)
        rows.append(np.concatenate([left.ravel(), right.ravel()]))
    return np.stack(rows)
