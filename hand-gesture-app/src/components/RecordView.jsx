import { useEffect, useRef, useState } from 'react'
import CameraView from './CameraView.jsx'
import { RECORD_SECONDS, createRecording, downloadRecording, frameFromResult } from '../lib/recording.js'
import { predictSign } from '../lib/api.js'

const COUNTDOWN_SECONDS = 3
// After the countdown, how long we wait for a hand to come into view before giving up.
const WAIT_FOR_HAND_SECONDS = 5

// "Record" mode: records the landmarks of one sign at a time.
// Flow: press Record (or Space) → 3-2-1 countdown → wait until a hand appears
// (so you can start with your hands down) → 3 seconds are captured → stop.
// The recording is then sent to the backend, and its answer is shown.
// It can also be downloaded as JSON for testing.
export default function RecordView({ model }) {
  const [ready, setReady] = useState(false)
  const [phase, setPhase] = useState('idle') // 'idle' | 'countdown' | 'waiting' | 'recording'
  const [countdown, setCountdown] = useState(COUNTDOWN_SECONDS)
  const [elapsed, setElapsed] = useState(0) // seconds recorded so far
  const [recording, setRecording] = useState(null) // the last finished recording
  const [notice, setNotice] = useState(null) // { type: 'ok' | 'error', text }
  // The backend's answer: { status: 'loading' } | { status: 'done', data } | { status: 'error', message }
  const [prediction, setPrediction] = useState(null)

  // The detection loop calls handleResult ~30 times a second. Values it needs
  // live in refs, because changing a ref doesn't re-render the page.
  const stageRef = useRef('off') // 'off' | 'waiting' | 'recording'
  const framesRef = useRef([])
  const startTimeRef = useRef(0) // timestamp of the first recorded frame

  function startRecording() {
    if (!ready || phase !== 'idle') return
    setNotice(null)
    setPrediction(null)
    setCountdown(COUNTDOWN_SECONDS)
    setPhase('countdown')
  }

  // Countdown: tick once a second, then wait for a hand.
  useEffect(() => {
    if (phase !== 'countdown') return
    if (countdown === 0) {
      framesRef.current = []
      stageRef.current = 'waiting'
      setElapsed(0)
      setPhase('waiting')
      return
    }
    const timer = setTimeout(() => setCountdown((c) => c - 1), 1000)
    return () => clearTimeout(timer)
  }, [phase, countdown])

  // Give up if no hand shows up in time.
  useEffect(() => {
    if (phase !== 'waiting') return
    const timer = setTimeout(() => {
      if (stageRef.current !== 'waiting') return
      stageRef.current = 'off'
      setPhase('idle')
      setNotice({ type: 'error', text: `No hand appeared within ${WAIT_FOR_HAND_SECONDS} seconds. Press Record to try again.` })
    }, WAIT_FOR_HAND_SECONDS * 1000)
    return () => clearTimeout(timer)
  }, [phase])

  function handleResult(result, { timestamp, width, height }) {
    if (stageRef.current === 'off') return

    // Waiting: start capturing on the first frame that has a hand in it.
    if (stageRef.current === 'waiting') {
      if (result.landmarks.length === 0) return
      stageRef.current = 'recording'
      startTimeRef.current = timestamp
      setPhase('recording')
    }

    const t = timestamp - startTimeRef.current // ms since the hand appeared
    framesRef.current.push(frameFromResult(result, t))
    setElapsed(Math.floor(t / 1000))

    // Stop after RECORD_SECONDS, however many frames the camera delivered in that time.
    if (t >= RECORD_SECONDS * 1000) {
      stageRef.current = 'off'
      setPhase('idle')
      const finished = createRecording(framesRef.current, width, height)
      if (finished) {
        setRecording(finished)
        setNotice({ type: 'ok', text: 'Recording finished.' })
        sendToBackend(finished)
      } else {
        setNotice({ type: 'error', text: 'No hand was visible in most frames, so the recording was discarded. Try again.' })
      }
    }
  }

  async function sendToBackend(rec) {
    setPrediction({ status: 'loading' })
    try {
      const data = await predictSign(rec)
      setPrediction({ status: 'done', data })
    } catch (err) {
      setPrediction({ status: 'error', message: err.message })
    }
  }

  // Space bar starts a recording, so your hands can stay in position.
  useEffect(() => {
    function onKeyDown(e) {
      if (e.code !== 'Space' || ['INPUT', 'SELECT', 'BUTTON'].includes(e.target.tagName)) return
      e.preventDefault()
      startRecording()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })

  const busy = phase !== 'idle'

  return (
    <section className="record">
      <CameraView model={model} onResult={handleResult} onReadyChange={setReady}>
        {phase === 'countdown' && <div className="camera-countdown">{countdown}</div>}
        {phase === 'waiting' && (
          <div className="camera-recording">
            <span className="rec-dot waiting" /> Start signing… recording begins when a hand appears
          </div>
        )}
        {phase === 'recording' && (
          <div className="camera-recording">
            <span className="rec-dot" /> Recording {elapsed}/{RECORD_SECONDS} s
          </div>
        )}
      </CameraView>

      <div className="panel">
        <button type="button" className="button primary" onClick={startRecording} disabled={!ready || busy}>
          {buttonLabel(phase, countdown)}
        </button>

        {notice && <p className={`notice ${notice.type}`}>{notice.text}</p>}

        {recording && (
          <p className="hint">
            {recording.sequenceLength} frames, hand visible in {recording.handFrames}.
          </p>
        )}

        {prediction && <PredictionResult prediction={prediction} />}

        <button
          type="button"
          className="button"
          onClick={() => downloadRecording(recording)}
          disabled={!recording || busy}
        >
          Download JSON
        </button>
      </div>
    </section>
  )
}

// The backend's answer: the recognised sign, its confidence, and the next best guesses.
function PredictionResult({ prediction }) {
  if (prediction.status === 'loading') return <p className="hint">Recognising…</p>
  if (prediction.status === 'error') return <p className="notice error">{prediction.message}</p>

  const { sign, confidence, top = [] } = prediction.data
  return (
    <div className="prediction">
      <span className="field-label">Recognised sign</span>
      <div className="prediction-sign">{sign}</div>
      <div className="prediction-confidence">{percent(confidence)} confidence</div>
      {top.length > 1 && (
        <ol className="prediction-top">
          {top.map((t) => (
            <li key={t.sign}>
              <span>{t.sign}</span>
              <span>{percent(t.confidence)}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

function percent(value) {
  return `${Math.round(value * 100)}%`
}

function buttonLabel(phase, countdown) {
  if (phase === 'countdown') return `Starting in ${countdown}…`
  if (phase === 'waiting') return 'Waiting for a hand…'
  if (phase === 'recording') return 'Recording…'
  return 'Record (Space)'
}
