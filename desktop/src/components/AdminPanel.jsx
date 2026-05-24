export default function AdminPanel({
  activeSection,
  loading,
  onClose,
  onRefresh,
  onSelectSection,
  onToggleTheme,
  open,
  sections,
  status,
  theme
}) {
  if (!open) return null;

  const currentSection = sections.find((section) => section.id === activeSection) || sections[0] || null;
  const basicSections = sections.filter((section) => section.group === "Setup" || section.id === "basics");
  const advancedGroups = sections
    .filter((section) => !basicSections.includes(section))
    .reduce((groups, section) => {
      const groupName = section.group || "Developer/operator tools";
      return {
        ...groups,
        [groupName]: [...(groups[groupName] || []), section]
      };
    }, {});

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
            <p>Start with Basics. Most users do not need the advanced operator panels.</p>
          </div>
          <div className="admin-header-actions">
            <span className={`status-pill status-pill-${status.tone}`}>{status.label}</span>
            <button
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              className="theme-toggle"
              onClick={onToggleTheme}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              type="button"
            >
              {theme === "dark" ? "Light" : "Dark"}
            </button>
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
            <p className="admin-nav-helper">Most users only need Basics. Advanced panels are for diagnostics and operator work.</p>
            {basicSections.map((section) => (
              <button
                className={section.id === currentSection?.id ? "active" : ""}
                key={section.id}
                onClick={() => onSelectSection(section.id)}
                type="button"
              >
                {section.label}
              </button>
            ))}
            <div className="admin-advanced-groups">
              {Object.entries(advancedGroups).map(([groupName, groupSections]) => (
                <details key={groupName} open={groupSections.some((section) => section.id === currentSection?.id)}>
                  <summary>{groupName}</summary>
                  <div className="admin-nav-group-buttons">
                    {groupSections.map((section) => (
                      <button
                        className={section.id === currentSection?.id ? "active" : ""}
                        key={section.id}
                        onClick={() => onSelectSection(section.id)}
                        type="button"
                      >
                        {section.label}
                      </button>
                    ))}
                  </div>
                </details>
              ))}
            </div>
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
