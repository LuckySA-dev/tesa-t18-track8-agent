"""Web monitor and control page for the t3-2 agent (Track 3-2): one page for the whole day.

    python t3-2/monitor.py              # http://127.0.0.1:8770
    python t3-2/monitor.py --port 8771

Reads the market with GETs only (state, calendar, leaderboard, version; cached, well under 5 req/s), the agent's
logs/decisions.jsonl and its memory file. Control (control.py): edit .env settings (keys write-only), check every
endpoint, start / stop the agent process (t3-2/main.py: the agent does all the trading), run pack.py.
There is NO code path that sends, edits or cancels an offer or an appraisal by hand (that is disqualification).
Keys are used only in Authorization headers and are never sent to the browser.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from appraisal import estimate_lots          # noqa: E402
from main import LOGS, load_env              # noqa: E402
from market import Market, MarketError       # noqa: E402
import control                               # noqa: E402

PAGE = HERE / "monitor.html"
# non-secret settings shown on the page (never TEAM_KEY / LLM_API_KEY)
SHOWN_ENV = ["MARKET_URL", "USE_LLM", "LLM_MODE", "LLM_MODEL", "LLM_BASE_URL", "LLM_CONTEXT", "LLM_MAX_TOKENS",
             "PATIENCE", "UNDERCUT", "K_FLOOR", "TV_MARGIN", "DYN_PR", "POS_IMPLIED_K", "JITTER", "LADDER",
             "GAP_PUSH", "SPEC_HOLD", "LATE_KEEP", "SAMPLE_NOISE", "N_TEAMS", "TACTIC", "TAC_RISK", "PLAN_BANS",
             "PATTERN", "LOG_DIR"]
FONTS = next((p for p in (HERE / "fonts", HERE.parent / "tesareplica" / "web" / "fonts") if p.is_dir()),
             HERE / "fonts")                                 # the market page's own Thai fonts
TTL = {"state": 2.0, "calendar": 60.0, "leaderboard": 10.0, "version": 60.0}


class Reader:
    """Cached GET-only access to the market. Market errors are returned as data, never raised."""

    def __init__(self, api: Market):
        self.api, self.cache, self.lock = api, {}, Lock()

    def get(self, name: str):
        with self.lock:
            hit = self.cache.get(name)
            if hit and time.time() - hit[0] < TTL[name]:
                return hit[1]
            try:
                out = {"ok": True, "data": getattr(self.api, name)()}
            except MarketError as e:
                out = {"ok": False, "error": f"{e.status} {e}"}
            except Exception as e:                       # noqa: BLE001 - network down etc.
                out = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            self.cache[name] = (time.time(), out)
            return out


def tail_events(n: int = 160) -> list[dict]:
    path = LOGS / "decisions.jsonl"
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        start = max(0, f.tell() - 400_000)
        f.seek(start)
        lines = f.read().decode("utf-8", "replace").splitlines()
        if start:                                        # first line may be cut in half
            lines = lines[1:]
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def memory_for(state: dict) -> dict:
    """The agent's memory file for this game (same key main.py uses: lot ids + samples)."""
    key = tuple(sorted((l["lot_id"], json.dumps(l["sample"], sort_keys=True)) for l in state.get("lots", [])))
    tag = hashlib.sha1(repr(key).encode()).hexdigest()[:8]
    path = LOGS / f"memory-{state.get('market')}-{tag}.json"
    if not path.exists():
        return {"file": path.name, "found": False, "sent": {}, "rounds": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    return {"file": path.name, "found": True, "sent": data.get("sent", {}), "rounds": data.get("rounds", [])}


def is_competition(state: dict | None) -> bool:
    """Scored market: port 8001, or the market itself says it is not the sandbox (any other URL or port)."""
    market = ((state or {}).get("data") or {}).get("market") if (state or {}).get("ok") else None
    return control.is_competition(os.environ.get("MARKET_URL", ""), market)


def snapshot(reader: Reader) -> dict:
    st, cal = reader.get("state"), reader.get("calendar")
    out = {"now": time.time(), "env": {k: os.environ.get(k) for k in SHOWN_ENV if os.environ.get(k) is not None},
           "state": st, "calendar": cal, "version": reader.get("version"), "leaderboard": reader.get("leaderboard"),
           "events": tail_events(), "log_dir": str(LOGS)}
    out["competition"] = is_competition(st)
    if st["ok"] and cal["ok"]:
        s, rules = st["data"], cal["data"].get("rules", {})
        try:
            tv = estimate_lots(rules, s["lots"])
        except Exception as e:                           # noqa: BLE001
            tv, out["tv_error"] = {}, repr(e)
        out["tv"] = tv
        total = sum(tv.get(l["lot_id"], 0) * l["kg_total"] for l in s["lots"])
        left = sum(tv.get(l["lot_id"], 0) * l["kg_remaining"] for l in s["lots"])
        rev = (s.get("standings") or {}).get("revenue_thb") or 0
        salvage = s.get("salvage_ratio") or rules.get("salvage_ratio") or 0.8
        out["pr_now"] = (rev + salvage * left) / total if total else None     # PR if the rest salvages now
        out["memory"] = memory_for(s)
    return out


def llm_check() -> dict:
    """One models + one tiny chat call to the configured gateway (only on button press)."""
    if os.environ.get("USE_LLM", "1") != "1":
        return {"ok": False, "error": "USE_LLM=0 (LLM off)"}
    from llm import LLM
    llm = LLM(os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"], os.environ.get("LLM_API_KEY", ""), timeout=20,
              max_tokens=20)
    t0 = time.time()
    try:
        models = llm.models()
        reply = llm.chat("Reply with the single word: ready")
        return {"ok": True, "models": str(models)[:300], "reply": str(reply)[:200], "seconds": round(time.time() - t0, 2)}
    except Exception as e:                               # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "seconds": round(time.time() - t0, 2)}


def make_handler(holder: dict):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, out, code: int = 200):
            self._send(code, json.dumps(out, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8")

        def _same_site(self) -> bool:
            """Only this page may POST: JSON content type (a cross-site page then needs a CORS preflight, which this
            server never grants) and, when the browser sends one, an Origin on this host and port."""
            if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                return False
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            o, port = urlparse(origin), self.server.server_address[1]
            return o.hostname in ("127.0.0.1", "localhost") and o.port == port

        def do_POST(self):                               # noqa: N802 - settings / checks / agent process / hand-in
            path = urlparse(self.path).path
            if not self._same_site():
                return self._send(403, b"forbidden: only the monitor page may change settings", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except ValueError:
                body = {}
            if path == "/api/config":
                out = control.save_config(body)
                holder["reader"] = Reader(Market(os.environ.get("MARKET_URL", ""), os.environ.get("TEAM_KEY", "")))
                return self._json(out)
            if path == "/api/check":
                return self._json(control.run_checks())
            if path == "/api/agent/start":                # never game after game in a scored market
                comp = is_competition(holder["reader"].get("state"))
                return self._json(control.agent_start(bool(body.get("forever")) and not comp))
            if path == "/api/agent/stop":
                return self._json(control.agent_stop())
            if path == "/api/pack":
                if body.get("send") and body.get("confirm") != "SEND":
                    return self._json({"ok": False, "output": "ส่งงานต้องยืนยันก่อน"})
                return self._json(control.pack(bool(body.get("send"))))
            self._send(404, b"not found", "text/plain")

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                                # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/fonts/") and (FONTS / Path(path).name).is_file():
                return self._send(200, (FONTS / Path(path).name).read_bytes(), "font/ttf")
            if path == "/api/config":
                return self._json(control.config_view())
            if path == "/api/agent":
                try:
                    after = int(parse_qs(urlparse(self.path).query).get("after", ["0"])[0])
                except ValueError:
                    after = 0
                return self._json(control.agent_status(after))
            if path == "/api/live":
                body = json.dumps(snapshot(holder["reader"]), ensure_ascii=False, default=str)
                return self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
            if path == "/api/llm":
                return self._send(200, json.dumps(llm_check(), ensure_ascii=False).encode("utf-8"),
                                  "application/json; charset=utf-8")
            self._send(404, b"not found", "text/plain")

        def log_message(self, *args):                    # keep the console quiet
            pass

    return Handler


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    load_env(HERE / ".env")
    holder = {"reader": Reader(Market(os.environ.get("MARKET_URL", ""), os.environ.get("TEAM_KEY", "")))}
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(holder))
    print(f"{datetime.now():%H:%M:%S} monitor http://127.0.0.1:{a.port}  market {os.environ.get('MARKET_URL')}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
