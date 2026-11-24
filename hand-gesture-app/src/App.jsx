import { useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import ThemeToggle from './components/ThemeToggle.jsx'
import LiveView from './components/LiveView.jsx'
import RecordView from './components/RecordView.jsx'
import { useTheme } from './hooks/useTheme.js'
import { useHandLandmarker } from './hooks/useHandLandmarker.js'

// The views a user can pick in the sidebar dropdown.
// To add a new one later, add an entry here and render it below.
const MODES = [
  { id: 'live', label: 'Live' },
  { id: 'record', label: 'Record' },
]

export default function App() {
  const [mode, setMode] = useState('live')
  const { theme, setTheme } = useTheme()
  // Loaded here, once, so switching modes doesn't reload the model.
  const model = useHandLandmarker()

  return (
    <div className="app">
      <Sidebar title="Hand Gesture Recognition" modes={MODES} mode={mode} onModeChange={setMode} />

      <main className="main">
        <header className="topbar">
          <ThemeToggle theme={theme} onChange={setTheme} />
        </header>

        {mode === 'live' && <LiveView model={model} />}
        {mode === 'record' && <RecordView model={model} />}
      </main>
    </div>
  )
}
