# Hand Gesture Recognition for Accessible Communication

A browser-based Auslan learning tool. A webcam captures someone signing, MediaPipe finds the hand landmarks in the browser, and an LSTM on a small Python backend recognises the sign and shows the English word with a confidence score.

Only landmark coordinates leave the browser, never video.

Group project for ITW601 Information Technology – Work Integrated Learning (Torrens University Australia).

## How it fits together

```
browser (hand-gesture-app)                         backend (FastAPI)
webcam → MediaPipe hand landmarks → 3 s recording ──POST /predict──→ preprocess → LSTM → { sign, confidence, top }
```

| Folder | What it is |
|---|---|
| [`hand-gesture-app/`](hand-gesture-app/README.md) | React + Vite front end: Live and Record modes |
| [`backend/`](backend/README.md) | Python 3.10: landmark extraction, preprocessing, training, API |
| [`backend/MODEL.md`](backend/MODEL.md) | The LSTM explained layer by layer, with results and references |
| `rgb/` | Video dataset, **not in the repository** (see below) |

## Quick start

Front end (Node.js):
```powershell
cd hand-gesture-app
npm install
npm run dev
```

Backend (Python 3.10, in a second terminal):
```powershell
py -3.10 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
cd backend
.venv\Scripts\activate
uvicorn main:app --port 8000
```

Open http://localhost:5173, choose **Record** and sign. The trained stage-1 model (10 signs) is in `backend/models/`, so prediction works without retraining.

## Dataset

Training videos are not included: they are large and not ours to redistribute. To retrain, put the videos in `rgb/` (stage 1 in `rgb/stage1/<SIGN>/` with `rgb/stage1/labels.json`), then run `extract_landmarks.py` and `train.py` as described in the [backend README](backend/README.md).

Stage 1 signs: HELLO, BYE-BYE, THANK YOU, WELCOME, FINE (OK), HELP, MOTHER, FATHER, TEACHER, WORK.
