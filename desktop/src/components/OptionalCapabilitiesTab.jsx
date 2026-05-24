function webSearchSummary(servicesStatus) {
  const services = Array.isArray(servicesStatus?.services) ? servicesStatus.services : [];
  const searxng = services.find((service) => service?.service_id === "searxng") || null;
  if (!searxng) {
    return {
      label: "Web search is optional",
      detail: "Ask the assistant to set it up when needed."
    };
  }
  if (searxng.enabled && searxng.configured && searxng.reachable) {
    return {
      label: "Web search is ready",
      detail: "SearXNG appears configured and reachable."
    };
  }
  return {
    label: "Web search is optional",
    detail: "Ask the assistant to set it up when needed. Safe web search uses SearXNG."
  };
}

export default function OptionalCapabilitiesTab({ onRefresh, servicesStatus }) {
  const webSearch = webSearchSummary(servicesStatus);
  const dockerAvailable = servicesStatus?.docker_available === true;
  const podmanAvailable = servicesStatus?.podman_available === true;

  return (
    <section className="grid">
      <div className="card">
        <h2>Optional capabilities</h2>
        <p className="help-text">These are optional local services. The assistant will show a plan and ask before changing anything.</p>
        <button onClick={onRefresh} type="button">Check optional services</button>
      </div>

      <div className="card">
        <div className="setup-card-head">
          <h3>Web search</h3>
          <span className={`badge ${webSearch.label.includes("ready") ? "health-ok" : "health-degraded"}`}>{webSearch.label}</span>
        </div>
        <p className="help-text">{webSearch.detail}</p>
        <p className="help-text">Search results are treated as untrusted summaries. The assistant will not open pages, download files, or install packs from search results without review.</p>
        <div className="setup-facts">
          <span>Docker: {dockerAvailable ? "found" : "not found"}</span>
          <span>Podman: {podmanAvailable ? "found" : "not found"}</span>
        </div>
      </div>
    </section>
  );
}
