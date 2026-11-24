import { useRef, useState } from 'react'
import CameraView from './CameraView.jsx'

// "Live" mode: the camera with landmarks and a count of detected hands.
export default function LiveView({ model }) {
  const [ready, setReady] = useState(false)
  const [handCount, setHandCount] = useState(0)
  const lastCount = useRef(0)

  function handleResult(result) {
    // Only update React state when the number changes, not 30 times a second.
    if (result.landmarks.length !== lastCount.current) {
      lastCount.current = result.landmarks.length
      setHandCount(lastCount.current)
    }
  }

  return (
    <section className="live">
      <CameraView model={model} onResult={handleResult} onReadyChange={setReady} />
      <p className="live-status">{ready ? `Hands detected: ${handCount}` : 'Waiting for camera and model…'}</p>
    </section>
  )
}
