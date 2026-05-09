#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-/tmp}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUTDIR/personal-agent-context-$STAMP.txt"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

json_or_raw() {
  if command -v python >/dev/null 2>&1; then
    python -m json.tool 2>/dev/null || cat
  else
    cat
  fi
}

section() {
  {
    echo
    echo "### $1"
  } >> "$OUT"
}

run() {
  {
    echo "\$ $*"
    "$@" 2>&1 || true
  } >> "$OUT"
}

curl_endpoint() {
  local name="$1"
  local url="$2"
  section "$name"
  {
    echo "\$ curl -sS $url"
    set +e
    curl -sS --max-time 10 "$url" 2>&1 | json_or_raw
    set -e
  } >> "$OUT"
}

redact_file() {
  python "$REPO_ROOT/scripts/redact_support_context.py" "$OUT" > "$OUT.redacted"
  mv "$OUT.redacted" "$OUT"
}

echo "Personal Agent support context" > "$OUT"
echo "Generated: $STAMP" >> "$OUT"
echo "Host: $(hostname)" >> "$OUT"

section "git"
run git branch --show-current
run git rev-parse --short HEAD
run git status --short
run git log --oneline -5

section "service: api"
run systemctl --user cat personal-agent-api.service
run systemctl --user status personal-agent-api.service --no-pager

section "service: telegram"
run systemctl --user cat personal-agent-telegram.service
run systemctl --user status personal-agent-telegram.service --no-pager

section "runtime split guidance"
cat >> "$OUT" <<'EOF'
If personal-agent-api.service runs from ~/.local/share/personal-agent/runtime/current, restarting that
service does not load repo checkout edits. Run bash scripts/promote_local_stable.sh after checkout
changes that should affect the stable API service.

If personal-agent-telegram.service runs from ~/personal-agent/.venv while the API service runs from
runtime/current, Telegram is using checkout/dev transport code and API is using stable code. This
split is acceptable only when Telegram ordinary chat is API-proxy-first through POST /chat and the
Telegram bridge smoke passes. Otherwise update the Telegram service to the stable runtime or promote
the checkout before restarting it.
EOF
run python -m agent split_status

curl_endpoint "live ready" "http://127.0.0.1:8765/ready"
curl_endpoint "live state" "http://127.0.0.1:8765/state"
curl_endpoint "llm status" "http://127.0.0.1:8765/llm/status"
curl_endpoint "llm catalog" "http://127.0.0.1:8765/llm/catalog"
curl_endpoint "telegram status" "http://127.0.0.1:8765/telegram/status"

section "ollama"
run ollama list

section "process"
run ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem

section "gpu"
run nvidia-smi

section "disk"
run df -h
run du -sh ~/.local/share/personal-agent 2>/dev/null

section "recent api logs"
run journalctl --user -u personal-agent-api.service -n 120 --no-pager

section "recent telegram logs"
run journalctl --user -u personal-agent-telegram.service -n 80 --no-pager

redact_file

echo "$OUT"
