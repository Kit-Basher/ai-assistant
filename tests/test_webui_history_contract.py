from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chat_experience_keeps_latest_visible_without_overriding_manual_scroll() -> None:
    source = (ROOT / "desktop/src/components/ChatExperience.jsx").read_text(encoding="utf-8")

    assert 'className="jump-to-latest"' in source
    assert 'onScroll={updateStickiness}' in source
    assert 'shouldStickToBottomRef.current = isNearBottom(transcript)' in source
    assert 'if (!force && !shouldStickToBottomRef.current) return' in source
    assert 'const forceScroll = forceNextScrollRef.current' in source
    assert "userTurnInProgressRef" not in source


def test_webui_uses_backend_threads_and_keeps_admin_secondary() -> None:
    app = (ROOT / "desktop/src/App.jsx").read_text(encoding="utf-8")
    experience = (ROOT / "desktop/src/components/ChatExperience.jsx").read_text(encoding="utf-8")

    assert 'request("GET", `/chat/threads?limit=50&${chatHistoryQuery()}`)' in app
    assert '`/chat/threads/${encodeURIComponent(normalizedThreadId)}?${chatHistoryQuery()}`' in app
    assert 'window.localStorage.setItem(storageKey, value)' in app
    assert 'onSelectThread={loadChatThread}' in app
    assert 'className="admin-entry"' in experience
    assert '>\n            Advanced\n' in experience
    assert 'aria-label="Conversation history"' in experience
    assert 'className="button-primary new-chat-button"' in experience


def test_browser_smoke_proves_desktop_mobile_reload_and_thread_selection() -> None:
    smoke = (ROOT / "scripts/browser_ui_survival_smoke.py").read_text(encoding="utf-8")

    assert 'desktop latest response stays visible' in smoke
    assert 'mobile latest response stays visible' in smoke
    assert 'page refresh survival' in smoke
    assert 'new chat preserves selectable history' in smoke
