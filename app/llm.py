"""Клиент LLM (OpenAI-совместимый chat/completions) с кэшем на диск и строгим парсингом.

Конфигурация — из окружения (поддерживается ../.env): API_KEY, OPENAI_URL, MODEL,
FALLBACK_MODEL. Значения секретов не логируются.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

_ENV_NAMES = ("API_KEY", "OPENAI_URL", "MODEL", "FALLBACK_MODEL")


def load_env(path: str | Path | None = None) -> None:
    """Загружает .env в os.environ (без перезаписи уже заданных переменных).

    Если путь не указан, .env ищется в текущей директории и выше.
    """
    candidates = [Path(path)] if path else [Path.cwd() / ".env", *Path.cwd().parents]
    for cand in candidates:
        env_file = cand if cand.name == ".env" else cand / ".env"
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return


class LLMClient:
    """Вызовы чат-модели; ответы кэшируются по хэшу (модель + системный + промпт)."""

    def __init__(self, cache_dir: str | Path = "models/llm_cache", timeout: int = 120,
                 retries: int = 3, temperature: float = 0.7) -> None:
        load_env()
        self.api_key = os.environ.get("API_KEY", "")
        self.url = os.environ.get("OPENAI_URL", "")
        self.model = os.environ.get("MODEL", "")
        self.fallback_model = os.environ.get("FALLBACK_MODEL", "")
        if not (self.api_key and self.url and self.model):
            raise RuntimeError("env API_KEY / OPENAI_URL / MODEL не заданы (см. ../.env)")
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def endpoint(self) -> str:
        """Полный URL chat/completions (OPENAI_URL может быть и базой, и полным адресом)."""
        url = self.url.rstrip("/")
        return url if url.endswith("/chat/completions") else f"{url}/chat/completions"

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def complete(self, prompt: str, system: str | None = None, use_fallback: bool = False) -> str:
        """Возвращает текст ответа модели (из кэша либо через API)."""
        model = self.fallback_model if (use_fallback and self.fallback_model) else self.model
        key = hashlib.sha256(f"{model}\n{system or ''}\n{prompt}\n{self.temperature}".encode()).hexdigest()
        cache = self._cache_path(key)
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))["response"]
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                payload = {"model": model, "temperature": self.temperature,
                           "messages": ([{"role": "system", "content": system}] if system else [])
                                       + [{"role": "user", "content": prompt}]}
                resp = requests.post(self.endpoint, json=payload, timeout=self.timeout,
                                     headers={"Authorization": f"Bearer {self.api_key}"})
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                cache.write_text(json.dumps({"response": text}, ensure_ascii=False), encoding="utf-8")
                return text
            except Exception as exc:  # noqa: BLE001 - сеть/JSON, ретраи выше
                last_err = exc
                time.sleep(2 * (attempt + 1))
        if not use_fallback and self.fallback_model:
            return self.complete(prompt, system, use_fallback=True)
        raise RuntimeError(f"LLM-запрос не удался: {type(last_err).__name__}")

    def complete_json(self, prompt: str, system: str | None = None) -> dict:
        """Вызов с ожиданием JSON-ответа; извлекает первый JSON-объект из вывода."""
        raw = self.complete(prompt, system)
        return _extract_json(raw)


def _extract_json(raw: str) -> dict:
    """Достаёт первый JSON-объект из ответа модели (терпит ```json-обёртку)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("в ответе LLM нет JSON-объекта")
    return json.loads(text[start : end + 1])
