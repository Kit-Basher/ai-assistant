from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
from pathlib import Path


class SecretStore:
    def __init__(self, path: str | None = None, service_name: str = "personal-agent") -> None:
        default_path = Path.home() / ".local" / "share" / "personal-agent" / "secrets.enc.json"
        self._path = Path(path) if path else default_path
        self._service_name = service_name
        self._keyring = self._load_keyring()

    @property
    def backend_name(self) -> str:
        if self._keyring is not None:
            return "keyring"
        return f"encrypted_file:{self._path}"

    @staticmethod
    def _load_keyring():
        try:
            import keyring  # type: ignore

            return keyring
        except Exception:
            return None

    @staticmethod
    def _machine_secret() -> bytes:
        machine_id_path = Path("/etc/machine-id")
        if machine_id_path.is_file():
            try:
                machine_id = machine_id_path.read_text(encoding="utf-8").strip()
                if machine_id:
                    return machine_id.encode("utf-8")
            except Exception:
                pass
        fallback = f"{platform.node()}:{os.getuid()}"
        return fallback.encode("utf-8")

    @classmethod
    def _derive_key(cls, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", cls._machine_secret(), salt, 200_000, dklen=32)

    @staticmethod
    def _xor_stream(data: bytes, key: bytes, nonce: bytes) -> bytes:
        output = bytearray()
        counter = 0
        while len(output) < len(data):
            block = hmac.new(key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
            take = min(len(block), len(data) - len(output))
            start = len(output)
            for idx in range(take):
                output.append(data[start + idx] ^ block[idx])
            counter += 1
        return bytes(output)

    @classmethod
    def _encrypt_payload(cls, payload: dict[str, str]) -> dict[str, str]:
        plaintext = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        salt = os.urandom(16)
        nonce = os.urandom(16)
        key = cls._derive_key(salt)
        ciphertext = cls._xor_stream(plaintext, key, nonce)
        mac = hmac.new(key, ciphertext, hashlib.sha256).digest()
        return {
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "mac": base64.b64encode(mac).decode("ascii"),
        }

    @classmethod
    def _decrypt_payload(cls, payload: dict[str, str]) -> dict[str, str]:
        salt = base64.b64decode(payload.get("salt", ""))
        nonce = base64.b64decode(payload.get("nonce", ""))
        ciphertext = base64.b64decode(payload.get("ciphertext", ""))
        mac = base64.b64decode(payload.get("mac", ""))
        key = cls._derive_key(salt)
        expected = hmac.new(key, ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, expected):
            raise RuntimeError("secret store integrity check failed")
        plaintext = cls._xor_stream(ciphertext, key, nonce)
        parsed = json.loads(plaintext.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError("invalid secret store payload")
        return {str(k): str(v) for k, v in parsed.items()}

    def _read_file_secrets(self) -> dict[str, str]:
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            return self._decrypt_payload(raw)
        except Exception:
            return {}

    def _write_file_secrets(self, data: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        encrypted = self._encrypt_payload(data)
        self._path.write_text(json.dumps(encrypted, ensure_ascii=True), encoding="utf-8")
        try:
            os.chmod(self._path, 0o600)
        except Exception:
            pass

    def get_secret(self, key: str) -> str | None:
        if self._keyring is not None:
            try:
                value = self._keyring.get_password(self._service_name, key)
                if value:
                    return str(value)
            except Exception:
                pass
        return self._read_file_secrets().get(key)

    def set_secret(self, key: str, value: str) -> None:
        cleaned = (value or "").strip()
        if self._keyring is not None:
            try:
                if cleaned:
                    self._keyring.set_password(self._service_name, key, cleaned)
                else:
                    self._keyring.delete_password(self._service_name, key)
                return
            except Exception:
                pass

        store = self._read_file_secrets()
        if cleaned:
            store[key] = cleaned
        else:
            store.pop(key, None)
        self._write_file_secrets(store)

    @staticmethod
    def _provider_key(provider: str) -> str:
        return f"provider:{provider}:api_key"

    def get_provider_api_key(self, provider: str) -> str | None:
        return self.get_secret(self._provider_key((provider or "").strip().lower()))

    def set_provider_api_key(self, provider: str, api_key: str) -> None:
        self.set_secret(self._provider_key((provider or "").strip().lower()), api_key)

    def configured_providers(self) -> list[str]:
        prefixes = "provider:"
        suffix = ":api_key"
        keys: list[str] = []

        if self._keyring is None:
            data = self._read_file_secrets()
            keys.extend(data.keys())

        providers: set[str] = set()
        for key in keys:
            if key.startswith(prefixes) and key.endswith(suffix):
                providers.add(key[len(prefixes) : -len(suffix)])
        return sorted(providers)
