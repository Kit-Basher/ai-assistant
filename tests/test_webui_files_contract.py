from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_webui_exposes_visible_first_class_local_files_contract() -> None:
    app = (ROOT / "desktop/src/App.jsx").read_text(encoding="utf-8")
    files_tab = (ROOT / "desktop/src/components/FilesTab.jsx").read_text(encoding="utf-8")
    vite = (ROOT / "desktop/vite.config.js").read_text(encoding="utf-8")

    assert 'label: "Files / Local Search"' in app
    assert "<FilesTab request={request} />" in app
    assert 'request("GET", "/filesystem/roots")' in files_tab
    assert "/filesystem/list?path=" in files_tab
    assert "/filesystem/read?path=" in files_tab
    assert "/filesystem/search?root=" in files_tab
    assert 'request("POST", "/filesystem/search_content"' in files_tab
    assert '"/filesystem": apiTarget' in vite
