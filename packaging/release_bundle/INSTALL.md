# Personal Agent Release Bundle Install

1. Extract the release bundle archive.
2. Run `bash install.sh`.
3. Open Personal Agent from the desktop menu or browse to `http://127.0.0.1:8765/`.

The installed service uses SAFE MODE and keeps Telegram disabled by default.
Mutable state is stored under `~/.local/share/personal-agent`; the versioned
runtime payload under `runtime/releases/` is replaceable code, not user state.

If you want to install into a custom user-local location, use:

- `bash install.sh --install-root /path/to/local/state`
