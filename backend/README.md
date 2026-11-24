# Backend

Python 3.10 + FastAPI. Receives a recording from the front end, prepares it, and runs the LSTM to recognise the sign.

## Set up (once)

From the repository folder (`Hand-gesture`):

```powershell
py -3.10 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Python 3.10 is needed because MediaPipe doesn't support the newest Python versions yet.

## 1. Extract landmarks from the training videos

```powershell
cd backend
.venv\Scripts\activate
python extract_landmarks.py --limit 2   # quick test: 2 videos per sign
python extract_landmarks.py             # all videos of the current stage
```

By default it uses the current training stage: `rgb/stage1/<SIGN>/` (10 signs × 12 videos: HELLO, BYE-BYE, THANK YOU, WELCOME, FINE (OK), HELP, MOTHER, FATHER, TEACHER, WORK) and `rgb/stage1/labels.json`. `--signs "HELLO,HELP"` limits it to some signs. For the full dataset use `--videos ../rgb --labels ../../Train.json --signs-file signs.txt` (one sign per line, names exactly as in `Train.json`). `--count` prints how many videos each sign has on disk without opening any. Each video becomes `data/landmarks/<videoId>.json` in the same format the front end sends. Already extracted videos are skipped, so you can stop and restart it. At the end it prints, per sign, how many videos it extracted and how many listed in `Train.json` are not in `rgb/`. MediaPipe prints a lot of `INFO` lines; they are normal.

## 2. Train the model

```powershell
python train.py             # about 1-3 minutes on a CPU
python train.py --seed 3    # another random split, to see how much accuracy varies
```

Splits each sign's videos into training (8-9), validation (1) and test (2), augments the training videos, trains the LSTM and prints test accuracy and a confusion matrix. Saves `models/sign_lstm.keras`, `models/labels.json` and `models/report.json`. Restart the API afterwards. **[MODEL.md](MODEL.md) explains the model layer by layer**, with results and references.

## 3. Run the API

```powershell
cd backend
.venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

- http://localhost:8000/docs lets you try the endpoints in the browser.
- `GET /health` reports whether a model is loaded.
- `POST /predict` takes a recording and returns `{ "sign", "confidence", "top": [{ "sign", "confidence" }, ...] }`. If `models/sign_lstm.keras` doesn't exist it answers `503`.

If the front end runs somewhere other than `localhost:5173`, set `ALLOWED_ORIGINS` (comma-separated) before starting the API.

## Files

- `preprocess.py`: turns any recording (video or webcam) into a 60 × 126 array: drops frames without hands at the start/end, resamples to 60 frames by timestamp, fixes the aspect ratio, centres each hand on its wrist and scales it by hand size. **Training and prediction must both use it.**
- `extract_landmarks.py`: runs MediaPipe Hand Landmarker (same `.task` model as the front end) over the videos.
- `dataset.py`: loads `data/landmarks/` for training through `preprocess.py`. Skips videos with fewer than 30 frames with a hand (`MIN_HAND_FRAMES`); `python dataset.py` lists what it keeps and skips.
- `train.py`: split, augmentation, LSTM, training and evaluation.
- `main.py`: the FastAPI app; loads the model from `models/` at start-up.
