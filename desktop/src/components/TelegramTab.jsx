export default function TelegramTab({
  saveTelegramToken,
  setTelegramToken,
  telegramConfigured,
  telegramStatus,
  telegramToken,
  testTelegramToken
}) {
  return (
    <section className="grid">
      <div className="card">
        <h2>Telegram Bot</h2>
        <p className="help-text">
          Status: {telegramConfigured ? "configured" : "not configured"}
        </p>
        <label>
          Bot token (stored securely)
          <input
            type="password"
            value={telegramToken}
            onChange={(event) => setTelegramToken(event.target.value)}
            placeholder="123456789:AA..."
          />
        </label>
        <div className="row-actions">
          <button className="button-primary" onClick={saveTelegramToken}>
            Save Token
          </button>
          <button onClick={testTelegramToken}>Test Telegram</button>
        </div>
        <p className="status-line">{telegramStatus || "No Telegram checks run yet."}</p>
      </div>
    </section>
  );
}
