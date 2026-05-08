#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-/tmp}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUTDIR/personal-agent-context-$STAMP.txt"

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
    curl -sS --max-time 10 "$url" 2>&1 | json_or_raw
  } >> "$OUT"
}

redact_file() {
  sed -E \
    -e 's/(token[=: ][[:space:]]*)[A-Za-z0-9:_-]{12,}/\1<redacted>/Ig' \
    -e 's/(api[_-]?key[=: ][[:space:]]*)[A-Za-z0-9._-]{12,}/\1<redacted>/Ig' \
    -e 's/(authorization: bearer )[A-Za-z0-9._-]+/\1<redacted>/Ig' \
    "$OUT" > "$OUT.redacted"
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
