// Talking to the Python backend (FastAPI).
// The address comes from the VITE_API_URL setting (see .env.example);
// without it we use the local backend.
export const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// Sends one recording and returns { sign, confidence, top: [{ sign, confidence }, ...] }.
// Throws an Error with a readable message if it fails.
export async function predictSign(recording) {
  let response
  try {
    response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(recording),
    })
  } catch {
    throw new Error(`Could not reach the backend at ${API_URL}. Is it running?`)
  }

  const body = await response.json().catch(() => null)
  if (!response.ok) {
    // FastAPI puts its error message in "detail".
    const detail = typeof body?.detail === 'string' ? body.detail : `HTTP ${response.status}`
    throw new Error(detail)
  }
  return body
}
