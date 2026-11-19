"""Ask an OpenAI-compatible LLM (vLLM gateway, Ollama, OpenAI) for a JSON answer.

ask_json() never raises: on timeout, a bad answer or any error it returns None and the
caller falls back to rules. Local LLMs fail often — the agent must survive that.
"""
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import httpx
from openai import OpenAI

# Gateway rules from the organisers (18 Nov): at most 2 sends per request, at most 20 calls a minute per team, no
# bursts of calls back to back. Defaults keep a margin; the counter is a file so main.py, check.py and monitor.py
# (same key) share it.
MAX_PER_MIN = int(os.environ.get("LLM_MAX_PER_MIN", "20"))
MIN_GAP_S = float(os.environ.get("LLM_MIN_GAP_S", "3"))
COOLDOWN_429_S = float(os.environ.get("LLM_COOLDOWN_429_S", "60"))
CALLS_FILE = Path(tempfile.gettempdir()) / "tesa-t18-llm-calls.json"


def calls_file(api_key: str) -> Path:
    """One counter per team KEY (the gateway counts per key): a practice agent on key-train must not use up the
    competition agent's key-test budget. Only a hash of the key is used in the file name."""
    tag = hashlib.sha256((api_key or "none").encode()).hexdigest()[:10]
    return CALLS_FILE.with_name(f"tesa-t18-llm-calls-{tag}.json")


class RateGuard:
    """Sliding-window limiter shared by every process on this PC. acquire() waits only briefly; if the next allowed
    call is further away it refuses, and the agent uses its rules for this round instead of losing round time."""

    def __init__(self, path: Path = CALLS_FILE, max_per_min: int = MAX_PER_MIN, min_gap: float = MIN_GAP_S,
                 clock=time.time, sleep=time.sleep):
        self.path, self.max_per_min, self.min_gap, self.clock, self.sleep = path, max_per_min, min_gap, clock, sleep

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:                                # noqa: BLE001 - missing / half-written file = no history
            return {}

    def _save(self, data: dict) -> None:
        try:
            self.path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass

    def wait_needed(self) -> float:
        d, now = self._load(), self.clock()
        calls = [t for t in d.get("t", []) if now - t < 60]
        wait = max(0.0, d.get("block_until", 0) - now)
        if calls:
            wait = max(wait, calls[-1] + self.min_gap - now)
        if len(calls) >= self.max_per_min:
            wait = max(wait, calls[-self.max_per_min] + 60 - now)
        return wait

    def acquire(self, max_wait: float) -> str | None:
        """None = go ahead (the call is recorded); otherwise the reason it was refused."""
        wait = self.wait_needed()
        if wait > max_wait:
            return f"rate guard: next LLM call allowed in {wait:.0f} s"
        if wait > 0:
            self.sleep(wait)
        d, now = self._load(), self.clock()
        d["t"] = [t for t in d.get("t", []) if now - t < 60] + [now]
        self._save(d)
        return None

    def cooldown(self, seconds: float = COOLDOWN_429_S) -> None:
        d = self._load()
        d["block_until"] = max(d.get("block_until", 0), self.clock() + seconds)
        self._save(d)

# Ways to switch Qwen's "thinking" on/off; servers disagree, so try each and remember what works.
THINKING_STYLES = [
    lambda on: {"reasoning_effort": "medium" if on else "none"},                    # Ollama, OpenAI
    lambda on: {"extra_body": {"chat_template_kwargs": {"enable_thinking": on}}},    # vLLM
]


class LLM:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 60, thinking: bool = False,
                 max_tokens: int = 1500, temperature: float = 0.7, connect_timeout: float = 3.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key or "none", timeout=timeout, max_retries=0)
        self.model, self.thinking, self.timeout = model, thinking, timeout
        self.connect_timeout = connect_timeout
        # LLM_TEMPERATURE (.env) overrides: lower = the same answer for the same facts, round after round
        self.max_tokens, self.temperature = max_tokens, float(os.environ.get("LLM_TEMPERATURE", temperature))
        self.style = 0
        self.last_error = None
        self.guard = RateGuard(path=calls_file(api_key))

    def _checked(self, call):
        """For the check / monitor helpers: respect the gateway limits, and start a cooldown on 429."""
        reason = self.guard.acquire(max_wait=self.guard.min_gap + 1)
        if reason:
            raise RuntimeError(reason)
        try:
            return call()
        except Exception as e:                           # noqa: BLE001
            if getattr(e, "status_code", None) == 429:
                self.guard.cooldown()
            raise

    def models(self) -> list[str]:
        """Model ids the server offers (GET /v1/models). Raises on connection errors: use it to test the gateway."""
        return self._checked(lambda: [m.id for m in self.client.models.list().data])

    def chat(self, user: str, system: str | None = None, max_tokens: int = 200) -> tuple[str, dict]:
        """Plain text answer + token usage, like the notebook's chat(). Raises on errors (for checks, not the agent)."""
        return self._checked(lambda: self._chat(user, system, max_tokens))

    def _chat(self, user: str, system: str | None, max_tokens: int) -> tuple[str, dict]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        r = self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens,
                                                temperature=self.temperature, **THINKING_STYLES[self.style](self.thinking))
        usage = r.usage.model_dump() if r.usage else {}
        return r.choices[0].message.content or "", usage

    def ask_json(self, system: str, user: str, schema: dict, timeout: float | None = None) -> dict | None:
        self.last_error = None
        # short connect timeout: a dead gateway must fail in seconds (it cost 21 s of a round on 18 Nov), while a
        # slow answer from a live gateway may still use the whole read budget
        total = min(timeout, self.timeout) if timeout else self.timeout
        client = self.client.with_options(timeout=httpx.Timeout(total, connect=min(self.connect_timeout, total)))
        t0 = time.time()
        for style in range(self.style, len(THINKING_STYLES)):   # at most 2 sends per request (gateway rule)
            reason = self.guard.acquire(max_wait=min(self.guard.min_gap + 1, total / 4))
            if reason:                                   # over the gateway limit: rules decide this round
                self.last_error = reason
                return None
            try:
                r = client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
                    **THINKING_STYLES[style](self.thinking))
            except Exception as e:                       # noqa: BLE001 — any failure means "use the fallback"
                self.last_error = f"{type(e).__name__}: {e}"[:300]
                if getattr(e, "status_code", None) == 429:   # over quota: no LLM for a while, never retry now
                    self.guard.cooldown()
                    return None
                if getattr(e, "status_code", None) == 400:   # server did not like this thinking switch
                    continue
                return None
            self.style = style
            text = r.choices[0].message.content or ""
            answer = parse_json(text)
            if answer is None:
                self.last_error = f"not JSON (finish={r.choices[0].finish_reason}): {text[:200]!r}"
            self.seconds = round(time.time() - t0, 1)
            return answer
        return None


def parse_json(text: str) -> dict | None:
    """Accept plain JSON, JSON in ``` fences, or JSON after a <think> block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None
