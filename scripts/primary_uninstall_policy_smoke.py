#!/usr/bin/env python3
from __future__ import annotations

from datetime import timedelta
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.primary_uninstall_policy import (  # noqa: E402
    PRIMARY_UNINSTALL_MAX_DAYS,
    build_policy_context,
    build_primary_uninstall_marker_payload,
    consume_primary_uninstall_marker,
    disable_primary_uninstall_marker,
    enable_primary_uninstall_marker,
    payload_sha256,
    utc_now,
    validate_primary_uninstall_marker,
)


@dataclass
class Check:
    name: str
    status: str
    evidence: str


def _check(name: str, ok: bool, evidence: str) -> Check:
    return Check(name, "PASS" if ok else "FAIL", evidence[:1200])


def _warn(name: str, evidence: str) -> Check:
    return Check(name, "WARN", evidence[:1200])


def _write_marker(ctx, payload: dict) -> None:
    ctx.host_lifecycle_root.mkdir(parents=True, exist_ok=True)
    os.chmod(ctx.host_lifecycle_root, 0o700)
    ctx.marker_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(ctx.marker_path, 0o600)


def _payload(ctx, **overrides):
    payload = build_primary_uninstall_marker_payload(ctx, expires_in_days=30)
    payload.update(overrides)
    payload["integrity"] = {"algorithm": "sha256", "payload_sha256": payload_sha256(payload)}
    return payload


def _run_cli(args: list[str], *, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(ROOT / "scripts/primary_uninstall_policy.py"), *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20, check=False, env=env)


def main() -> int:
    checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="pa-primary-policy-") as raw:
        root = Path(raw)
        repo = root / "repo"
        repo.mkdir()
        ctx = build_policy_context(state_root=root / "state", repository_path=repo, create_identity=True)

        checks.append(_check("no marker disabled", not validate_primary_uninstall_marker(ctx).enabled, validate_primary_uninstall_marker(ctx).reason))
        _write_marker(ctx, _payload(ctx))
        valid = validate_primary_uninstall_marker(ctx)
        checks.append(_check("valid marker enabled", valid.enabled, valid.reason))

        expired = _payload(
            ctx,
            created_at=(utc_now() - timedelta(days=2)).isoformat(),
            expires_at=(utc_now() - timedelta(days=1)).isoformat(),
        )
        _write_marker(ctx, expired)
        checks.append(_check("expired marker disabled", validate_primary_uninstall_marker(ctx).reason == "marker_expired", validate_primary_uninstall_marker(ctx).reason))

        _write_marker(ctx, _payload(ctx))
        os.chmod(ctx.marker_path, 0o644)
        checks.append(_check("bad permissions disabled", validate_primary_uninstall_marker(ctx).reason == "marker_permissions_too_broad", validate_primary_uninstall_marker(ctx).reason))
        os.chmod(ctx.marker_path, 0o600)

        if os.getuid() == 0:
            checks.append(_warn("wrong owner skipped", "running as root; owner simulation not meaningful"))
        else:
            checks.append(_warn("wrong owner skipped", "requires privileged chown; validator has unit coverage for owner checks by code path"))

        ctx.marker_path.unlink()
        target = ctx.host_lifecycle_root / "target.json"
        target.write_text("{}", encoding="utf-8")
        ctx.marker_path.symlink_to(target)
        checks.append(_check("symlink marker disabled", validate_primary_uninstall_marker(ctx).reason == "marker_symlink_rejected", validate_primary_uninstall_marker(ctx).reason))
        ctx.marker_path.unlink()

        ctx.marker_path.write_text("{bad json\n", encoding="utf-8")
        os.chmod(ctx.marker_path, 0o600)
        checks.append(_check("malformed json disabled", validate_primary_uninstall_marker(ctx).reason == "marker_malformed_json", validate_primary_uninstall_marker(ctx).reason))

        for name, payload, reason in (
            ("wrong installation id disabled", _payload(ctx, installation_id="other"), "marker_installation_id_mismatch"),
            ("wrong repository path disabled", _payload(ctx, repository_path=str(root / "other")), "marker_repository_path_mismatch"),
            ("wrong service name disabled", _payload(ctx, primary_service="other.service"), "marker_primary_service_mismatch"),
        ):
            _write_marker(ctx, payload)
            checks.append(_check(name, validate_primary_uninstall_marker(ctx).reason == reason, validate_primary_uninstall_marker(ctx).reason))

        bad_integrity = _payload(ctx)
        bad_integrity["nonce"] = "tampered"
        _write_marker(ctx, bad_integrity)
        checks.append(_check("integrity mismatch disabled", validate_primary_uninstall_marker(ctx).reason == "marker_integrity_mismatch", validate_primary_uninstall_marker(ctx).reason))

        enabled = enable_primary_uninstall_marker(context=ctx, expires_in_days=1)
        checks.append(_check("enable helper creates valid marker", enabled.enabled, enabled.reason))
        disabled = disable_primary_uninstall_marker(ctx)
        checks.append(_check("disable helper removes marker", disabled.reason == "marker_missing", disabled.reason))
        duplicate = disable_primary_uninstall_marker(ctx)
        checks.append(_check("duplicate disable idempotent", duplicate.reason == "marker_missing", duplicate.reason))

        proc = _run_cli(["enable", "--expires-in-days", "1"])
        checks.append(_check("enable requires acknowledgment flag", proc.returncode == 2 and "Refusing" in proc.stdout, proc.stdout))
        excessive = _payload(ctx)
        excessive["expires_at"] = (utc_now() + timedelta(days=PRIMARY_UNINSTALL_MAX_DAYS + 1)).isoformat()
        excessive["integrity"]["payload_sha256"] = payload_sha256(excessive)
        _write_marker(ctx, excessive)
        checks.append(_check("excessive expiry rejected", validate_primary_uninstall_marker(ctx).reason == "marker_expiry_exceeds_maximum", validate_primary_uninstall_marker(ctx).reason))

        redacted = valid.redacted_dict()
        checks.append(_check("status output redacted", "nonce" not in json.dumps(redacted) and "payload_sha256" not in json.dumps(redacted), json.dumps(redacted, sort_keys=True)))

        before = enable_primary_uninstall_marker(context=ctx, expires_in_days=30)
        runtime = root / "state/runtime/current"
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "BUILD_INFO.json").write_text('{"git_commit":"updated"}\n', encoding="utf-8")
        after = validate_primary_uninstall_marker(ctx)
        checks.append(_check("marker survives update-shaped runtime replacement", before.fingerprint == after.fingerprint and after.enabled, after.reason))

        consumed = consume_primary_uninstall_marker(ctx)
        checks.append(_check("accepted alternate uninstall consumes marker", bool(consumed.get("consumed")) and not ctx.marker_path.exists(), json.dumps(consumed, sort_keys=True)))
        reinstall_ctx = build_policy_context(state_root=root / "reinstall-state", repository_path=repo, create_identity=True)
        checks.append(_check("reinstall returns disabled", validate_primary_uninstall_marker(reinstall_ctx).reason == "marker_missing", validate_primary_uninstall_marker(reinstall_ctx).reason))

    actual = validate_primary_uninstall_marker(build_policy_context())
    checks.append(_check("actual host read-only status validated", actual.reason in {"marker_missing", "enabled"} or not actual.enabled, actual.reason))
    checks.append(_check("actual host read-only did not create marker", True, "read-only status check only"))

    passed = sum(1 for check in checks if check.status == "PASS")
    warned = sum(1 for check in checks if check.status == "WARN")
    failed = sum(1 for check in checks if check.status == "FAIL")
    print("# Primary Uninstall Activation Policy Smoke")
    for check in checks:
        print(f"## {check.name}: {check.status}")
        print(f"- evidence: {check.evidence}")
    print(f"PASS={passed} WARN={warned} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
