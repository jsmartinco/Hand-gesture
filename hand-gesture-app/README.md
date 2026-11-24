# Hand Gesture Recognition

Browser front end for the ITW601 Auslan learning project. It shows the webcam feed and draws hand landmarks detected by MediaPipe.

Modes (sidebar dropdown):
- **Live**: camera with landmarks and a hand count.
- **Record**: records one sign. Press **Record** or **Space**, wait for the 3-2-1 countdown and sign. You can start with your hands down: capture begins on the first frame where a hand is visible (it gives up after 5 seconds without one) and stops 3 seconds later (about 90 frames at 30 fps; the backend resamples every recording to 60 frames). A recording with a hand in fewer than a third of its frames is thrown away. A finished recording is sent to the backend's `POST /predict` and the answer (sign, confidence, best guesses) is shown under the buttons. **Download JSON** saves the last recording.

The backend address comes from `VITE_API_URL` (default `http://localhost:8000`). To change it, copy `.env.example` to `.env.local` and restart `npm run dev`.

## Recording file

```json
{
  "version": 2,
  "source": "webcam",
  "recordedAt": "2026-11-19T10:15:30.000Z",
  "imageWidth": 1280,
  "imageHeight": 720,
  "sequenceLength": 90,
  "handFrames": 58,
  "landmarkFormat": "…",
  "frames": [ { "t": 0, "left": [[x, y, z], … 21 points] or null, "right": … }, … one per camera frame for 3 s ]
}
```

`t` is milliseconds since the first frame. `imageWidth`/`imageHeight` are the camera size in pixels. No video is saved, only landmark coordinates. `left`/`right` is the handedness MediaPipe reports on the unmirrored camera image. The backend's `extract_landmarks.py` writes the training videos in this same format.

## Run it

```bash
npm install
npm run dev
```

Open http://localhost:5173 and allow camera access.

## Structure

```
src/
  main.jsx                  entry point, mounts <App />
  App.jsx                   page layout, selected mode, theme
  index.css                 all styles and the light/dark colour variables
  components/
    Sidebar.jsx             title + mode dropdown
    ThemeToggle.jsx         sun / moon buttons
    CameraView.jsx          webcam + landmark drawing, shared by all modes
    LiveView.jsx            Live mode
    RecordView.jsx          Record mode
  lib/
    recording.js            recording format and JSON download
    api.js                  sends a recording to the backend
  hooks/
    useTheme.js             remembers light/dark choice
    useHandLandmarker.js    loads the MediaPipe hand model
public/
  models/hand_landmarker.task   MediaPipe hand model
  mediapipe/wasm/               MediaPipe runtime (copied on npm install)
scripts/copy-mediapipe-wasm.js
```
