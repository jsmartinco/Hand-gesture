import { useEffect, useState } from 'react'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'

// Both paths are served from public/ by Vite (see scripts/copy-mediapipe-wasm.js).
const WASM_PATH = '/mediapipe/wasm'
const MODEL_PATH = '/models/hand_landmarker.task'

// Loads MediaPipe's HandLandmarker once and returns it when ready.
// status: 'loading' | 'ready' | 'error'
export function useHandLandmarker() {
  const [landmarker, setLandmarker] = useState(null)
  const [status, setStatus] = useState('loading')
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    let instance = null

    async function load() {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_PATH)
        instance = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 2,
        })
        if (cancelled) {
          instance.close()
          return
        }
        setLandmarker(instance)
        setStatus('ready')
      } catch (err) {
        if (cancelled) return
        setError(err)
        setStatus('error')
      }
    }

    load()

    // Runs when the component using this hook goes away: free the model's memory.
    return () => {
      cancelled = true
      instance?.close()
    }
  }, [])

  return { landmarker, status, error }
}
