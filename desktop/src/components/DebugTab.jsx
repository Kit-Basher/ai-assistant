export default function DebugTab({ logs }) {
  return (
    <section className="card debug-card">
      <h2>Recent Requests</h2>
      <div className="debug-list">
        {logs.length === 0 ? <p className="empty">No requests yet.</p> : null}
        {logs.map((row, index) => (
          <div key={`${row.time}-${index}`} className={`debug-item ${row.ok ? "ok" : "error"}`}>
            <div className="debug-head">
              <span>{row.time}</span>
              <span>{row.endpoint}</span>
              <span>{row.ok ? "OK" : "ERROR"}</span>
            </div>
            <p>{row.detail}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
