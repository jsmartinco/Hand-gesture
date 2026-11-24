"""Trains the LSTM that recognises a sign from 60 frames of hand landmarks.

Run (from the backend folder, virtual environment active):
    python train.py
    python train.py --seed 7          # a different random split, to see how much accuracy varies

Steps:
  1. Load data/landmarks/*.json through dataset.py (same preprocessing as the API).
  2. Split per sign: 2 videos for the test set, 1 for validation, the rest for training.
     The test videos are never seen during training, so the test accuracy is honest.
  3. Augment the training videos (rotate, scale, noise, change speed, mirror) to make
     many slightly different copies, because 8-9 videos per sign is very little.
  4. Train the LSTM, stopping when the validation loss stops improving.
  5. Report accuracy and a confusion matrix on the test set, and save:
       models/sign_lstm.keras   the trained model (main.py loads it)
       models/labels.json       sign names in the order of the model's outputs
       models/report.json       split, settings, accuracy and confusion matrix

MODEL.md explains every layer and choice in detail.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's start-up messages

import numpy as np

from dataset import MIN_HAND_FRAMES, load_dataset
from preprocess import FEATURES_PER_FRAME, NUM_POINTS, SEQUENCE_LENGTH

HERE = Path(__file__).parent
MODELS = HERE / "models"

TEST_PER_SIGN = 2
VAL_PER_SIGN = 1


# --- 1. Split -------------------------------------------------------------

def split_per_sign(y, rng):
    """Returns index arrays (train, val, test), the same number of test/val videos for every sign."""
    train, val, test = [], [], []
    for sign in np.unique(y):
        idx = rng.permutation(np.where(y == sign)[0])
        test += list(idx[:TEST_PER_SIGN])
        val += list(idx[TEST_PER_SIGN : TEST_PER_SIGN + VAL_PER_SIGN])
        train += list(idx[TEST_PER_SIGN + VAL_PER_SIGN :])
    return np.array(train), np.array(val), np.array(test)


# --- 2. Augmentation ------------------------------------------------------
# Works on preprocessed arrays (60, 126). Each frame is reshaped to
# (2 hands, 21 points, 3 coordinates). A hand that wasn't visible is all zeros
# and must stay all zeros, so every change is multiplied by a "present" mask.

def augment(sequence, rng):
    s = sequence.reshape(SEQUENCE_LENGTH, 2, NUM_POINTS, 3).copy()
    present = np.any(s != 0, axis=(2, 3), keepdims=True)  # (60, 2, 1, 1): is this hand visible?

    # Speed and timing: take a slightly shorter part of the sign (up to 10% off each
    # end) and stretch it back to 60 frames, so the sign looks faster or slower.
    start = rng.uniform(0, 0.1) * (SEQUENCE_LENGTH - 1)
    end = (1 - rng.uniform(0, 0.1)) * (SEQUENCE_LENGTH - 1)
    frames = np.round(np.linspace(start, end, SEQUENCE_LENGTH)).astype(int)
    s, present = s[frames], present[frames]

    # Rotation in the image plane (camera or signer slightly tilted): ±15 degrees.
    angle = np.radians(rng.uniform(-15, 15))
    cos, sin = np.cos(angle), np.sin(angle)
    x, y = s[..., 0].copy(), s[..., 1].copy()
    s[..., 0] = cos * x - sin * y
    s[..., 1] = sin * x + cos * y

    # Size: hands and finger proportions vary between people: ±10%.
    s *= rng.uniform(0.9, 1.1)

    # Noise: MediaPipe's points jitter a little from frame to frame.
    s += rng.normal(0, 0.02, s.shape)

    # Mirror half of the time: what a left-handed signer looks like.
    # Flip x and swap the left and right hand.
    if rng.random() < 0.5:
        s[..., 0] *= -1
        s = s[:, ::-1]
        present = present[:, ::-1]

    s *= present  # missing hands back to exactly zero
    return s.reshape(SEQUENCE_LENGTH, FEATURES_PER_FRAME).astype(np.float32)


def augmented_set(X, y, copies, rng):
    """The original videos plus `copies` augmented versions of each."""
    new_X, new_y = [X], [y]
    for _ in range(copies):
        new_X.append(np.stack([augment(seq, rng) for seq in X]))
        new_y.append(y)
    return np.concatenate(new_X), np.concatenate(new_y)


# --- 3. Model -------------------------------------------------------------

def build_model(num_signs):
    """The LSTM. See MODEL.md for why each layer is there."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # One recording: 60 time steps, 126 numbers each (2 hands x 21 points x 3).
        keras.Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)),
        # Frames where no hand is visible (all 126 values are 0) are skipped by the LSTMs.
        layers.Masking(mask_value=0.0),
        # First LSTM: reads the frames in order and outputs 64 numbers per frame
        # describing the movement so far (return_sequences=True keeps all 60 steps).
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        # Second LSTM: reads those 60 steps and summarises the whole sign in 32 numbers.
        layers.LSTM(32),
        layers.Dropout(0.3),
        # A small fully connected layer that combines the summary.
        layers.Dense(32, activation="relu"),
        # One output per sign; softmax turns them into probabilities that add up to 1.
        layers.Dense(num_signs, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",  # labels are numbers 0..9, not one-hot vectors
        metrics=["accuracy"],
    )
    return model


# --- 4. Train, evaluate, save ---------------------------------------------

def confusion_matrix(true, predicted, n):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(true, predicted):
        m[t, p] += 1
    return m


def print_confusion(matrix, labels):
    short = [l[:6] for l in labels]
    print("\nConfusion matrix (rows = true sign, columns = predicted):")
    print(" " * 12 + " ".join(f"{s:>6}" for s in short))
    for label, row in zip(labels, matrix):
        print(f"{label[:11]:<12}" + " ".join(f"{v:>6}" for v in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split, augmentation and weights")
    parser.add_argument("--copies", type=int, default=20, help="augmented copies of each training video")
    parser.add_argument("--epochs", type=int, default=300, help="maximum epochs (early stopping usually ends sooner)")
    parser.add_argument("--patience", type=int, default=30, help="epochs without validation improvement before stopping")
    args = parser.parse_args()

    from tensorflow import keras
    keras.utils.set_random_seed(args.seed)  # same seed -> same split and same result
    rng = np.random.default_rng(args.seed)

    X, y, labels, ids, skipped = load_dataset()
    print(f"{len(X)} videos, {len(labels)} signs ({len(skipped)} skipped: fewer than {MIN_HAND_FRAMES} hand frames)")

    train, val, test = split_per_sign(y, rng)
    X_train, y_train = augmented_set(X[train], y[train], args.copies, rng)
    print(f"train {len(train)} videos -> {len(X_train)} with augmentation, val {len(val)}, test {len(test)}")

    model = build_model(len(labels))
    model.summary()

    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X[val], y[val]),
        epochs=args.epochs,
        batch_size=32,
        shuffle=True,
        callbacks=[stop],
        verbose=2,
    )

    probabilities = model.predict(X[test], verbose=0)
    predicted = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predicted == y[test]))
    top3 = float(np.mean([y[test][i] in probabilities[i].argsort()[-3:] for i in range(len(test))]))
    matrix = confusion_matrix(y[test], predicted, len(labels))

    print(f"\nStopped after {len(history.history['loss'])} epochs (best at {stop.best_epoch + 1}).")
    print(f"Test accuracy: {accuracy:.0%} ({int(np.sum(predicted == y[test]))}/{len(test)} videos)   top-3: {top3:.0%}")
    print_confusion(matrix, labels)

    MODELS.mkdir(exist_ok=True)
    model.save(MODELS / "sign_lstm.keras")
    (MODELS / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    report = {
        "seed": args.seed,
        "settings": {"copies": args.copies, "maxEpochs": args.epochs, "patience": args.patience,
                     "minHandFrames": MIN_HAND_FRAMES, "testPerSign": TEST_PER_SIGN, "valPerSign": VAL_PER_SIGN},
        "videos": {"train": len(train), "val": len(val), "test": len(test), "skipped": [s[0] for s in skipped]},
        "testVideoIds": [ids[i] for i in test],
        "epochs": len(history.history["loss"]),
        "bestEpoch": stop.best_epoch + 1,
        "testAccuracy": accuracy,
        "testTop3Accuracy": top3,
        "labels": labels,
        "confusionMatrix": matrix.tolist(),
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()},
    }
    (MODELS / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved model, labels and report to {MODELS}")


if __name__ == "__main__":
    main()
