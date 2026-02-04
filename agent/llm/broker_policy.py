from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderPolicy:
    id: str
    provider: str
    remote: bool
    model: str
    capabilities: list[str]
    cost: int
    latency: int
    reliability: int


@dataclass
class BrokerPolicy:
    providers: list[ProviderPolicy]
    weights: dict[str, int]
    tie_breaker: list[str]


def load_policy(path: str) -> BrokerPolicy:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    data = _parse_policy_text(text)
    return _validate_policy(data)


def _parse_policy_text(text: str) -> dict[str, Any]:
    providers: list[dict[str, Any]] = []
    weights: dict[str, Any] = {}
    selection: dict[str, Any] = {}

    section = None
    current_provider: dict[str, Any] | None = None

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and ":" in line:
            key = line.split(":", 1)[0].strip()
            section = key
            if section == "providers":
                continue
            if section == "weights":
                continue
            if section == "selection":
                continue
            raise ValueError(f"Unknown policy section: {section}")

        if section == "providers":
            stripped = line.lstrip()
            if stripped.startswith("- "):
                if current_provider:
                    providers.append(current_provider)
                current_provider = {}
                rest = stripped[2:].strip()
                if rest:
                    key, value = _parse_key_value(rest)
                    current_provider[key] = _parse_value(value)
                continue
            if current_provider is None:
                raise ValueError("Provider entry must start with '-'")
            key, value = _parse_key_value(stripped)
            current_provider[key] = _parse_value(value)
            continue

        if section == "weights":
            key, value = _parse_key_value(line.strip())
            weights[key] = _parse_value(value)
            continue

        if section == "selection":
            key, value = _parse_key_value(line.strip())
            selection[key] = _parse_value(value)
            continue

        raise ValueError("Missing policy section header.")

    if current_provider:
        providers.append(current_provider)

    return {"providers": providers, "weights": weights, "selection": selection}


def _parse_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"Invalid line: {line}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_value(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",")]
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    return value.strip().strip("'\"")


def _validate_policy(data: dict[str, Any]) -> BrokerPolicy:
    providers_raw = data.get("providers")
    weights = data.get("weights")
    selection = data.get("selection")

    if not isinstance(providers_raw, list) or not providers_raw:
        raise ValueError("Policy must include providers list.")
    if not isinstance(weights, dict):
        raise ValueError("Policy must include weights map.")
    if not isinstance(selection, dict):
        raise ValueError("Policy must include selection map.")

    providers: list[ProviderPolicy] = []
    for entry in providers_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each provider must be a mapping.")
        try:
            provider = ProviderPolicy(
                id=str(entry["id"]),
                provider=str(entry["provider"]),
                remote=bool(entry["remote"]),
                model=str(entry["model"]),
                capabilities=list(entry["capabilities"]),
                cost=int(entry["cost"]),
                latency=int(entry["latency"]),
                reliability=int(entry["reliability"]),
            )
        except KeyError as exc:
            raise ValueError(f"Provider missing required field: {exc}") from exc
        providers.append(provider)

    required_weights = {"cost", "latency", "reliability"}
    for key in required_weights:
        if key not in weights:
            raise ValueError(f"Missing weight: {key}")
    weights_int = {k: int(v) for k, v in weights.items()}

    tie_breaker = selection.get("tie_breaker")
    if not isinstance(tie_breaker, list) or not tie_breaker:
        raise ValueError("selection.tie_breaker must be a list.")

    return BrokerPolicy(providers=providers, weights=weights_int, tie_breaker=[str(x) for x in tie_breaker])
