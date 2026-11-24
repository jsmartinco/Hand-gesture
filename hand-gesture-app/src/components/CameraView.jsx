import { useEffect, useRef, useState } from 'react'
import { DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision'

// The camera square, shared by every mode: shows the webcam feed and draws the
// hand landmarks that MediaPipe finds on a <canvas> laid exactly over the video.
//
// Props:
//   model         { landmarker, status, error } from useHandLandmarker (loaded once in App)
//   onResult      called for every processed frame with (MediaPipe result, { timestamp, width, height })
//   onReadyChange called with true/false when camera + model become (un)ready
//   children      extra things drawn on top of the video (e.g. a countdown)
export default function CameraView({ model, onResult, onReadyChange, children }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [cameraStatus, setCameraStatus] = useState('starting') // 'starting' | 'on' | 'error'
  const [cameraError, setCameraError] = useState(null)
  const { landmarker, status: modelStatus, error: modelError } = model

  // Keep the latest callbacks in refs, so the detection loop below doesn't
  // have to restart every time the parent re-renders.
  const onResultRef = useRef(onResult)
  const onReadyChangeRef = useRef(onReadyChange)
  useEffect(() => {
    onResultRef.current = onResult
    onReadyChangeRef.current = onReadyChange
  })

  // 1. Turn the webcam on, and off again when this view is closed.
  useEffect(() => {
    let stream = null
    let cancelled = false

    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop())
          return
        }
        stream = s
        videoRef.current.srcObject = s
        setCameraStatus('on')
      })
      .catch((err) => {
        if (cancelled) return
        setCameraError(err)
        setCameraStatus('error')
      })

    return () => {
      cancelled = true
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const ready = Boolean(landmarker) && cameraStatus === 'on'

  // Tell the parent when we become ready (or stop being ready).
  useEffect(() => {
    onReadyChangeRef.current?.(ready)
  }, [ready])

  // 2. Once both the camera and the model are ready, run detection on every new frame.
  useEffect(() => {
    if (!ready) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const drawer = new DrawingUtils(ctx)
    let frameId = 0
    let lastVideoTime = -1

    function loop() {
      // Only process a frame when the video has actually advanced.
      if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime

        // Match the canvas resolution to the camera's so the dots line up.
        if (canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        }

        const timestamp = performance.now()
        const result = landmarker.detectForVideo(video, timestamp)

        ctx.clearRect(0, 0, canvas.width, canvas.height)
        for (const hand of result.landmarks) {
          drawer.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: '#2dd4bf', lineWidth: 4 })
          drawer.drawLandmarks(hand, { color: '#f97316', lineWidth: 1, radius: 4 })
        }

        // Also pass when the frame was taken (ms) and the camera image size in pixels.
        onResultRef.current?.(result, { timestamp, width: video.videoWidth, height: video.videoHeight })
      }
      frameId = requestAnimationFrame(loop)
    }

    frameId = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(frameId)
  }, [ready, landmarker])

  const message = statusMessage(cameraStatus, cameraError, modelStatus, modelError)

  return (
    <div className="camera-frame">
      <video ref={videoRef} className="camera-video" autoPlay playsInline muted />
      <canvas ref={canvasRef} className="camera-overlay" />
      {message ? <div className="camera-message">{message}</div> : children}
    </div>
  )
}

function statusMessage(cameraStatus, cameraError, modelStatus, modelError) {
  if (cameraStatus === 'error') {
    return cameraError?.name === 'NotAllowedError'
      ? 'Camera access was blocked. Allow it in the browser address bar and reload.'
      : `Could not start the camera (${cameraError?.message ?? 'unknown error'}).`
  }
  if (modelStatus === 'error') return `Could not load the hand model (${modelError?.message ?? 'unknown error'}).`
  if (cameraStatus === 'starting') return 'Starting camera…'
  if (modelStatus === 'loading') return 'Loading hand model…'
  return null
}
