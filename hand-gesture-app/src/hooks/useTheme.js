import { useEffect, useState } from 'react'

const STORAGE_KEY = 'theme'

// Starts from the saved choice, or the operating system's setting the first time.
function initialTheme() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    // localStorage can be blocked (private mode); fall through to the OS setting.
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

// Keeps the theme in state and writes it to <html data-theme="...">,
// which the CSS variables in index.css read.
export function useTheme() {
  const [theme, setTheme] = useState(initialTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      // Not saving is fine; the theme still applies for this visit.
    }
  }, [theme])

  return { theme, setTheme }
}
