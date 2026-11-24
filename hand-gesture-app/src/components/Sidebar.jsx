// Left-hand bar: app title and the dropdown that picks which view is shown.
export default function Sidebar({ title, modes, mode, onModeChange }) {
  return (
    <aside className="sidebar">
      <h1 className="sidebar-title">{title}</h1>

      <label className="field">
        <span className="field-label">Mode</span>
        <select value={mode} onChange={(e) => onModeChange(e.target.value)}>
          {modes.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </label>
    </aside>
  )
}
