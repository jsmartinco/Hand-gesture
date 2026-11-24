// What one recording looks like, and saving it as a JSON file.
// A recording is the hand landmarks of one sign, sent to the backend (LSTM)
// to be recognised. Only coordinates are kept, never video.
//
// The backend's extract_landmarks.py writes the training videos in this same
// format, so backend/preprocess.py treats webcam recordings and videos identically.

// How long a recording lasts, counted from the first frame with a hand.
// 3 seconds covers about 90% of the training videos' signs. The number of
// frames depends on the camera's speed (about 90 at 30 fps); the backend
// resamples every recording to the same length anyway.
export const RECORD_SECONDS = 3

// A recording is kept only if at least this share of its frames contains a hand
// (a third of 3 seconds = about 1 second of hand, the same rule as in training).
const MIN_HAND_FRAMES = 1 / 3

const FORMAT_VERSION = 2

// Turns one MediaPipe result into a frame:
//   { t: ms since the first frame, left: [[x, y, z] × 21] | null, right: ... }
// x and y are 0–1 across the image (x of the width, y of the height), z is depth
// relative to the wrist. "left"/"right" is the handedness as MediaPipe reports it
// on the unmirrored camera image.
export function frameFromResult(result, t) {
  const frame = { t: Math.round(t), left: null, right: null }
  result.landmarks.forEach((hand, i) => {
    const points = hand.map((p) => [round(p.x), round(p.y), round(p.z)])
    let side = result.handedness[i]?.[0]?.categoryName === 'Left' ? 'left' : 'right'
    // MediaPipe occasionally labels both hands the same; keep both anyway.
    if (frame[side]) side = side === 'left' ? 'right' : 'left'
    frame[side] = points
  })
  return frame
}

// Returns the recording object, or null if too few frames had a hand in them.
// width/height are the camera image size in pixels; the backend needs them to
// undo the stretching of x and y when the image isn't square.
export function createRecording(frames, width, height) {
  const handFrames = frames.filter((f) => f.left || f.right).length
  if (handFrames < frames.length * MIN_HAND_FRAMES) return null
  return {
    version: FORMAT_VERSION,
    source: 'webcam',
    recordedAt: new Date().toISOString(),
    imageWidth: width,
    imageHeight: height,
    sequenceLength: frames.length,
    handFrames,
    landmarkFormat: '21 MediaPipe hand landmarks per hand, [x, y, z], x/y normalised 0-1; t in ms',
    frames,
  }
}

export function downloadRecording(recording) {
  const blob = new Blob([JSON.stringify(recording)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  // e.g. recording-2026-11-19T10-15-30.json
  link.download = `recording-${recording.recordedAt.slice(0, 19).replaceAll(':', '-')}.json`
  link.click()
  URL.revokeObjectURL(url)
}

function round(n) {
  return Math.round(n * 10000) / 10000
}
