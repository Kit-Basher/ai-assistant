export default function AdminPanel({
  activeSection,
  loading,
  onClose,
  onRefresh,
  onSelectSection,
  open,
  sections,
  status
}) {
  if (!open) return null;

  const currentSection = sections.find((section) => section.id === activeSection) || sections[0] || null;

  return (
    <div className="admin-overlay" onClick={onClose} role="presentation">
      <aside
        aria-label="Advanced admin tools"
        className="admin-drawer"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="admin-header">
          <div>
            <p className="product-kicker">Advanced</p>
            <h2>Admin tools</h2>
            <p>Setup, diagnostics, provider controls, and operator-only workflows live here.</p>
          </div>
          <div className="admin-header-actions">
            <span className={`status-pill status-pill-${status.tone}`}>{status.label}</span>
            <button onClick={onRefresh} type="button">
              Refresh
            </button>
            <button className="button-primary" onClick={onClose} type="button">
              Done
            </button>
          </div>
        </div>

        <div className="admin-body">
          <nav className="admin-nav">
            {sections.map((section) => (
              <button
                className={section.id === currentSection?.id ? "active" : ""}
                key={section.id}
                onClick={() => onSelectSection(section.id)}
                type="button"
              >
                {section.label}
              </button>
            ))}
          </nav>

          <div className="admin-content">
            {loading ? (
              <div className="admin-loading">
                <div className="admin-loading-card">
                  <p className="inline-action-eyebrow">Loading</p>
                  <h3>Fetching advanced state</h3>
                  <p>This area loads the full operator surface only when you open it.</p>
                </div>
              </div>
            ) : (
              currentSection?.content || null
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}
