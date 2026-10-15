"""Provider-agnostic structured-JSON client for the AI analyst.

The user picks the model in .env, so the analyst cannot depend on one vendor: AI_PROVIDER=anthropic talks to
Claude through the official async SDK (streaming, schema-constrained output, server-side refusal fallback),
AI_PROVIDER=openai talks to any OpenAI-compatible /chat/completions endpoint (OpenAI, OpenRouter, Groq, Gemini's
OpenAI endpoint, a local Ollama/vLLM). Either way the caller gets one parsed JSON object that has been checked
against the schema — with one repair round-trip when a model returns something malformed — or an LLMError
whose message can be shown to the user as-is.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

from ..config import ROOT, settings

log = logging.getLogger("td.research.llm")

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
# models that accept the server-side refusal fallback (`fallbacks: "default"` under this beta header)
FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"})
FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
OPENAI_URL = "https://api.openai.com/v1"
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """A model call that produced no usable answer. `kind` tells callers whether tokens were spent:
    config / auth / model / network / timeout / rate / status (nothing billed) vs refusal / truncated / invalid."""

    def __init__(self, message: str, kind: str = "status"):
        super().__init__(message)
        self.kind = kind

    @property
    def billable(self) -> bool:
        return self.kind in ("refusal", "truncated", "invalid")


@dataclass
class LLMResult:
    data: dict
    provider: str
    model: str
    usage: dict | None
    stop_reason: str | None


@dataclass
class _Raw:
    """One model reply before validation."""
    text: str
    usage: dict | None
    stop_reason: str | None
    model: str
    data: Any = None
    error: str | None = None
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ JSON helpers
_TYPES: dict[str, tuple[type, ...]] = {"object": (dict,), "array": (list,), "string": (str,),
                                       "integer": (int,), "number": (int, float), "boolean": (bool,)}


def validate(data: Any, schema: dict, path: str = "$", errors: list[str] | None = None) -> list[str]:
    """Check `data` against the subset of JSON Schema the report uses (type, enum, properties, required, items).
    Extra object keys are tolerated here and dropped by `prune`. Returns human-readable error strings."""
    errs = [] if errors is None else errors
    if len(errs) >= 25:
        return errs
    t = schema.get("type")
    if t:
        ok = isinstance(data, _TYPES.get(t, (object,)))
        if t in ("integer", "number") and isinstance(data, bool):
            ok = False
        if ok and t == "number" and isinstance(data, float) and data != data:
            ok = False
        if not ok:
            errs.append(f"{path}: expected {t}, got {type(data).__name__}")
            return errs
    if "enum" in schema and data not in schema["enum"]:
        errs.append(f"{path}: {data!r} is not one of {schema['enum']}")
    if isinstance(data, dict):
        for key in schema.get("required", []):
            if key not in data:
                errs.append(f"{path}: missing required key '{key}'")
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                validate(data[key], sub, f"{path}.{key}", errs)
    elif isinstance(data, list) and "items" in schema:
        for i, item in enumerate(data):
            validate(item, schema["items"], f"{path}[{i}]", errs)
    return errs


def prune(data: Any, schema: dict) -> Any:
    """Drop keys the schema does not declare (models without strict mode sometimes add their own)."""
    if isinstance(data, dict) and "properties" in schema:
        props = schema["properties"]
        return {k: prune(v, props[k]) for k, v in data.items() if k in props}
    if isinstance(data, list) and "items" in schema:
        return [prune(x, schema["items"]) for x in data]
    return data


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def extract_json(text: str) -> Any:
    """Parse the model's reply: plain JSON, a ```json fence, or the first balanced {...} object in prose
    (local reasoning models often wrap the answer in <think> blocks or commentary). Raises ValueError."""
    s = (text or "").strip()
    if not s:
        raise ValueError("empty reply")
    try:
        return json.loads(s)
    except ValueError:
        pass
    s = _THINK_RE.sub("", s).strip()
    for m in _FENCE_RE.finditer(s):
        try:
            return json.loads(m.group(1).strip())
        except ValueError:
            continue
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s, i)
                return obj
            except ValueError:
                continue
    raise ValueError("no JSON object in the reply")


def _merge_usage(a: dict | None, b: dict | None) -> dict | None:
    if not a or not b:
        return a or b
    return {k: None if a.get(k) is None and b.get(k) is None else (a.get(k) or 0) + (b.get(k) or 0)
            for k in {**a, **b}}


def _env_lookup(names: tuple[str, ...], env: Mapping[str, str] | None) -> str:
    """First non-empty value among `names` in the process environment or, failing that, the project's .env
    (pydantic-settings reads .env for its own fields only, so ANTHROPIC_API_KEY there would otherwise be missed)."""
    if env is not None:
        return next((env[n].strip() for n in names if (env.get(n) or "").strip()), "")
    for n in names:
        if (os.environ.get(n) or "").strip():
            return os.environ[n].strip()
    try:
        from dotenv import dotenv_values
        vals = dotenv_values(ROOT / ".env")
    except Exception:
        return ""
    return next((str(vals[n]).strip() for n in names if (vals.get(n) or "").strip()), "")


# ------------------------------------------------------------------ client
class LLMClient:
    """`await LLMClient().json(system, user, schema, name)` → LLMResult with a schema-checked dict."""

    def __init__(self, cfg=settings, anthropic_client=None, http_transport: httpx.AsyncBaseTransport | None = None,
                 env: Mapping[str, str] | None = None):
        self.cfg = cfg
        self.provider = str(cfg.ai_provider or "anthropic").lower()
        self._anthropic = anthropic_client
        self._anthropic_loop: asyncio.AbstractEventLoop | None = None
        self._transport = http_transport
        self.retry_pause = 2.0            # seconds before retrying a 429/5xx on the OpenAI-compatible path
        fallback = ("ANTHROPIC_API_KEY",) if self.provider == "anthropic" else ("OPENAI_API_KEY",)
        self.key = (cfg.ai_api_key or "").strip() or _env_lookup(fallback, env)
        self.model = (cfg.ai_model or "").strip()
        if self.provider == "anthropic":
            self.model = self.model or DEFAULT_ANTHROPIC_MODEL
            ok = bool(self.key) or anthropic_client is not None
            self.reason = "" if ok else (
                "No AI key configured. Set AI_API_KEY (or ANTHROPIC_API_KEY) in .env and restart to add the "
                f"written analysis; AI_MODEL defaults to {DEFAULT_ANTHROPIC_MODEL}. Any OpenAI-compatible model "
                "works too with AI_PROVIDER=openai.")
        elif self.provider == "openai":
            base = (cfg.ai_base_url or "").strip()
            if not self.model:
                ok, self.reason = False, ("AI_PROVIDER=openai needs AI_MODEL (e.g. the model name your provider "
                                          "or local server lists) in .env.")
            elif not self.key and not base:
                ok, self.reason = False, ("No AI key configured. Set AI_API_KEY (or OPENAI_API_KEY) in .env, or "
                                          "point AI_BASE_URL at an OpenAI-compatible server that needs no key.")
            else:
                ok, self.reason = True, ""
        else:
            ok, self.reason = False, f"Unknown AI_PROVIDER {self.provider!r}: use anthropic or openai."
        self.enabled = ok

    # ---- public
    async def json(self, system: str, user: str, schema: dict, name: str) -> LLMResult:
        if not self.enabled:
            raise LLMError(self.reason, "config")
        raw = await self._call(system, user, schema, name)
        errs = self._check(raw, schema)
        if errs:
            log.warning("%s %s output failed validation (%d issues) — asking for a repair", self.provider,
                        self.model, len(errs))
            prev = raw.text if raw.data is None else json.dumps(raw.data, ensure_ascii=False)
            repair = (f"{user}\n\nYour previous output failed validation: {'; '.join(errs[:15])}.\n"
                      f"Previous output:\n{prev[:30000]}\n\nReturn only the corrected JSON object.")
            first_usage = raw.usage
            raw = await self._call(system, repair, schema, name)
            raw.usage = _merge_usage(first_usage, raw.usage)
            errs = self._check(raw, schema)
            if errs:
                raise LLMError(f"the model's answer did not match the report format twice ({errs[0]})", "invalid")
        return LLMResult(data=prune(raw.data, schema), provider=self.provider, model=raw.model or self.model,
                         usage=raw.usage, stop_reason=raw.stop_reason)

    @staticmethod
    def _check(raw: _Raw, schema: dict) -> list[str]:
        if raw.data is None:
            try:
                raw.data = extract_json(raw.text)
            except ValueError as e:
                return [f"reply is not valid JSON ({e})"]
        return validate(raw.data, schema)

    async def _call(self, system: str, user: str, schema: dict, name: str) -> _Raw:
        if self.provider == "anthropic":
            return await self._anthropic_call(system, user, schema)
        return await self._openai_call(system, user, schema, name)

    # ---- Anthropic (official SDK)
    def _anthropic_client(self):
        if self._anthropic is not None and self._anthropic_loop is None:
            return self._anthropic                       # injected by the caller
        loop = asyncio.get_running_loop()
        if self._anthropic is None or self._anthropic_loop is not loop:
            import anthropic
            # base_url only when AI_BASE_URL is set: otherwise the SDK's own default (or ANTHROPIC_BASE_URL) applies
            self._anthropic = anthropic.AsyncAnthropic(api_key=self.key, timeout=self.cfg.ai_timeout, max_retries=2,
                                                       base_url=(self.cfg.ai_base_url or "").strip() or None)
            self._anthropic_loop = loop
        return self._anthropic

    async def _anthropic_call(self, system: str, user: str, schema: dict) -> _Raw:
        import anthropic
        client = self._anthropic_client()
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        effort = (self.cfg.ai_effort or "").strip().lower()
        if effort in EFFORTS:
            output_config["effort"] = effort
        elif effort:
            log.warning("ignoring AI_EFFORT=%r (use one of %s)", effort, ", ".join(sorted(EFFORTS)))
        # thinking and sampling params are deliberately not sent: current models think adaptively by default
        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=int(self.cfg.ai_max_tokens),
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}], output_config=output_config)
        try:
            if self.model in FALLBACK_MODELS:
                manager = client.beta.messages.stream(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                manager = client.messages.stream(**kwargs)
            async with manager as stream:
                msg = await stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise LLMError("AI key rejected by Anthropic (401) — check AI_API_KEY", "auth") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"this API key may not use {self.model} (403): {_sdk_msg(e)}", "auth") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"unknown AI_MODEL {self.model!r} (404) — check the model id", "model") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Anthropic rate limit reached (429) — try again in a minute", "rate") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"Anthropic rejected the request (400): {_sdk_msg(e)}", "status") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {_sdk_msg(e)}", "status") from e
        except anthropic.APITimeoutError as e:
            raise LLMError(f"the model did not answer within AI_TIMEOUT={self.cfg.ai_timeout:g}s", "timeout") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("could not reach the Anthropic API (network error)", "network") from e
        except anthropic.AnthropicError as e:
            raise LLMError(f"Anthropic client error: {str(e)[:300]}", "status") from e

        usage = None
        if getattr(msg, "usage", None) is not None:
            u = msg.usage
            usage = {"input_tokens": int(getattr(u, "input_tokens", 0) or 0),
                     "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
                     "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None)}
        stop = getattr(msg, "stop_reason", None)
        if stop == "refusal":
            details = getattr(msg, "stop_details", None)
            cat = getattr(details, "category", None) if details is not None else None
            raise LLMError(f"the model declined this request (refusal{f', category {cat}' if cat else ''})",
                           "refusal")
        if stop == "max_tokens":
            raise LLMError(f"output cut off at AI_MAX_TOKENS={self.cfg.ai_max_tokens} — raise AI_MAX_TOKENS",
                           "truncated")
        # after a mid-stream server-side fallback the partial text and the fallback model's continuation are
        # separate text blocks around a `fallback` marker; together they form the answer
        blocks = [b for b in (getattr(msg, "content", None) or []) if getattr(b, "type", None) == "text"]
        if not blocks:
            raise LLMError("the model returned no text", "invalid")
        text = "".join(getattr(b, "text", "") or "" for b in blocks)
        raw = _Raw(text=text, usage=usage, stop_reason=stop, model=str(getattr(msg, "model", "") or self.model))
        try:
            raw.data = json.loads(text)
        except ValueError:
            try:
                raw.data = json.loads(blocks[-1].text)       # continuation alone, if the join was not valid
            except (ValueError, AttributeError):
                raw.data = None
        return raw

    # ---- OpenAI-compatible
    async def _openai_call(self, system: str, user: str, schema: dict, name: str) -> _Raw:
        base = ((self.cfg.ai_base_url or "").strip() or OPENAI_URL).rstrip("/")
        url = f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        formats: list[dict | None] = [
            {"type": "json_schema", "json_schema": {"name": re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "output",
                                                    "schema": schema, "strict": True}},
            {"type": "json_object"},
            None]
        fi, retries = 0, 0
        token_key: str | None = "max_tokens"
        async with httpx.AsyncClient(transport=self._transport, timeout=self.cfg.ai_timeout) as client:
            while True:
                body: dict[str, Any] = {"model": self.model, "messages": [
                    {"role": "system", "content": system}, {"role": "user", "content": user}]}
                if token_key:
                    body[token_key] = int(self.cfg.ai_max_tokens)
                if formats[fi] is not None:
                    body["response_format"] = formats[fi]
                if formats[fi] is None or formats[fi]["type"] == "json_object":
                    # json_object mode requires the word JSON in the messages; no schema is enforced server-side
                    body["messages"][1]["content"] = (f"{user}\n\nReply with one JSON object only, matching this "
                                                      f"JSON schema:\n{json.dumps(schema)}")
                try:
                    resp = await client.post(url, json=body, headers=headers)
                except httpx.TimeoutException as e:
                    raise LLMError(f"the model did not answer within AI_TIMEOUT={self.cfg.ai_timeout:g}s",
                                   "timeout") from e
                except httpx.HTTPError as e:
                    raise LLMError(f"could not reach {base} ({type(e).__name__})", "network") from e
                if resp.status_code == 400:
                    msg = _http_msg(resp)
                    low = msg.lower()
                    if token_key == "max_tokens" and "max_tokens" in low and (
                            "max_completion_tokens" in low or "unsupported" in low):
                        token_key = "max_completion_tokens"
                        continue
                    if token_key and token_key in low and any(
                            w in low for w in ("too large", "maximum", "at most", "exceed", "less than or equal")):
                        log.info("%s caps output below AI_MAX_TOKENS=%s — using the provider's default", base,
                                 self.cfg.ai_max_tokens)
                        token_key = None
                        continue
                    if formats[fi] is not None and fi < 2 and any(
                            w in low for w in ("response_format", "json_schema", "json_object", "structured")):
                        fi += 1
                        log.info("%s does not accept this response_format — retrying with %s", base,
                                 (formats[fi] or {}).get("type", "plain text"))
                        continue
                    raise LLMError(f"the AI provider rejected the request (400): {msg[:300]}", "status")
                if resp.status_code in (401, 403):
                    raise LLMError(f"AI key rejected by {base} ({resp.status_code}) — check AI_API_KEY", "auth")
                if resp.status_code == 404:
                    raise LLMError(f"unknown AI_MODEL {self.model!r} or wrong AI_BASE_URL (404 from {url})", "model")
                if resp.status_code in _RETRY_STATUS and retries < 2:
                    retries += 1
                    pause = self.retry_pause * 2 ** (retries - 1)
                    try:
                        pause = max(pause, float(resp.headers.get("retry-after", 0)))
                    except ValueError:
                        pass
                    await asyncio.sleep(min(pause, 30.0))
                    continue
                if resp.status_code == 429:
                    raise LLMError(f"rate limit reached at {base} (429) — try again later", "rate")
                if resp.status_code >= 400:
                    raise LLMError(f"AI provider error {resp.status_code}: {_http_msg(resp)[:300]}", "status")
                break
        try:
            payload = resp.json()
            choice = payload["choices"][0]
            message = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError("the AI provider returned an unexpected response (no choices)", "status") from e
        u = payload.get("usage") or {}
        usage = {"input_tokens": int(u.get("prompt_tokens") or 0),
                 "output_tokens": int(u.get("completion_tokens") or 0),
                 "cache_read_input_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")} if u else None
        finish = choice.get("finish_reason")
        if message.get("refusal"):
            raise LLMError(f"the model declined this request: {str(message['refusal'])[:200]}", "refusal")
        if finish == "content_filter":
            raise LLMError("the provider's content filter blocked the answer", "refusal")
        if finish == "length":
            raise LLMError(f"output cut off at AI_MAX_TOKENS={self.cfg.ai_max_tokens} — raise AI_MAX_TOKENS",
                           "truncated")
        content = message.get("content")
        if isinstance(content, list):          # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not content:
            raise LLMError("the model returned no text", "invalid")
        return _Raw(text=str(content), usage=usage, stop_reason=finish,
                    model=str(payload.get("model") or self.model))


def _sdk_msg(e: Exception) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:300]
    return str(getattr(e, "message", e))[:300]


def _http_msg(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:500]
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)
    if isinstance(err, str):
        return err
    return json.dumps(body)[:500]
