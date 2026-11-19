"""Control side of the monitor page (monitor.py): settings in .env, endpoint checks, the agent process, hand-in.

Nothing here sends, edits or cancels an offer or an appraisal: trading stays with the agent (t3-2/main.py), which
this module only starts and stops. Keys are write-only: the page can set TEAM_KEY / LLM_API_KEY, never read them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / ".env"
SECRETS = {"TEAM_KEY", "LLM_API_KEY"}
# settings the page may change: (key, label, kind, choices / hint)
EDITABLE = [
    ("MARKET_URL", "ที่อยู่ตลาด", "text", "แข่ง: http://202.29.227.22:8001 · ฝึก: :8000 (ไม่มี /trade ต่อท้าย)"),
    ("TEAM_KEY", "key ของทีม", "secret", "แข่ง = Key-test · ฝึก = Key-train"),
    ("STRATEGY", "กลยุทธ์", "choice", ["adaptive", ""]),
    ("TACTIC", "tactic", "choice", ["1", "0"]),
    ("SPEC_P", "ความมั่นใจสเปก", "text", "0.5 (ค่าที่ทดสอบแล้ว)"),
    ("FREE_TRY", "ลองล็อตเกือบผ่าน", "choice", ["1", "0"]),
    ("USE_LLM", "ใช้ LLM", "choice", ["0", "1"]),
    ("LLM_BASE_URL", "LLM gateway", "text", "แข่งใช้ได้เฉพาะ http://202.29.227.22:4000/v1"),
    ("LLM_MODEL", "LLM model", "text", "event-llm"),
    ("LLM_API_KEY", "LLM key", "secret", "key เดียวกับตลาด"),
    ("PATTERN", "pattern (ด่าน 2 พัง)", "choice", ["0", "1"]),
]
_PROC: dict = {"proc": None, "lines": deque(maxlen=400), "seq": 0, "started": None, "args": None}
_LOCK = threading.Lock()


# ------------------------------------------------------------------ settings (.env)
def read_env() -> dict:
    out = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def config_view() -> dict:
    env = read_env()
    rows = []
    for key, label, kind, extra in EDITABLE:
        v = env.get(key, "")
        row = {"key": key, "label": label, "kind": kind,
               "choices": extra if isinstance(extra, list) else None, "hint": extra if isinstance(extra, str) else ""}
        if key in SECRETS:
            row["set"], row["length"] = bool(v), len(v)
        else:
            row["value"] = v
            live = os.environ.get(key)
            if live is not None and live != v:      # a shell variable overrides .env for this page and its agent
                row["hint"] = f"ตอนนี้ใช้ {live or '(ว่าง)'} จากตัวแปรระบบ (ทับค่าใน .env)" + (" · " + row["hint"] if row["hint"] else "")
        rows.append(row)
    return {"rows": rows, "file": str(ENV_FILE), "log_dir": os.environ.get("LOG_DIR") or "t3-2/logs (ค่าปกติ)"}


def save_config(values: dict) -> dict:
    """Write the given keys into .env (comments and order kept), and into this process's environment."""
    allowed = {k for k, *_ in EDITABLE}
    changes = {k: str(v).strip() for k, v in (values or {}).items() if k in allowed}
    bad = sorted(k for k, v in changes.items() if any(c in v for c in "\r\n\0"))
    if bad:                                  # a newline would write extra KEY=VALUE lines into .env
        return {"saved": [], "note": f"ไม่ได้บันทึก: ค่าของ {', '.join(bad)} มีการขึ้นบรรทัดใหม่", "error": True}
    changes = {k: v for k, v in changes.items() if not (k in SECRETS and v == "")}   # empty secret = keep
    if "MARKET_URL" in changes:
        u = urlparse(changes["MARKET_URL"])
        changes["MARKET_URL"] = f"{u.scheme}://{u.netloc}" if u.scheme and u.netloc else changes["MARKET_URL"]
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    done = set()
    for i, line in enumerate(lines):
        if "=" in line and not line.lstrip().startswith("#"):
            k = line.split("=", 1)[0].strip()
            if k in changes:
                lines[i] = f"{k}={changes[k]}"
                done.add(k)
    lines += [f"{k}={v}" for k, v in changes.items() if k not in done]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in changes.items():
        os.environ[k] = v
    return {"saved": sorted(changes), "note": "agent ที่รันอยู่ยังใช้ค่าเดิมจนกว่าจะหยุดแล้วรันใหม่"}


# ------------------------------------------------------------------ checks
def _get(url: str, key: str | None = None, timeout: float = 6.0):
    """GET with the market's 5 requests / s rule in mind: space calls, and wait + retry once after a 429."""
    for attempt in range(2):
        time.sleep(0.3)
        out = _get_once(url, key, timeout)
        if out[0] != 429:
            return out
        wait = ((out[1] or {}).get("error", {}).get("allowed", {}) or {}).get("retry_after_s", 1.0) if isinstance(out[1], dict) else 1.0
        time.sleep(min(3.0, float(wait or 1.0)) + 0.2)
    return out


def _get_once(url: str, key: str | None, timeout: float):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null"), round((time.time() - t0) * 1000)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"null")
        except ValueError:
            body = None
        return e.code, body, round((time.time() - t0) * 1000)
    except Exception as e:                                   # noqa: BLE001 - reported, not raised
        return None, repr(e), round((time.time() - t0) * 1000)


def other_agents() -> list[int]:
    """PIDs of t3-2 main.py processes on this PC that this page did not start (two agents = double offers)."""
    if os.name != "nt":
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'main\\.py' -and "
                              "$_.CommandLine -match 't3-2' } | ForEach-Object { $_.ProcessId }"],
                             capture_output=True, text=True, timeout=15).stdout
        mine = _PROC["proc"].pid if running() else None
        return [int(x) for x in out.split() if x.strip().isdigit() and int(x) != mine]
    except Exception:                                        # noqa: BLE001
        return []


def is_competition(url: str, market: str | None = None) -> bool:
    """Scored market: port 8001, or GET /v1/state says the market is not the sandbox (whatever URL or port)."""
    return urlparse(url or "").port == 8001 or market not in (None, "", "sandbox")


def practice_in_hand_in(hand_in: Path) -> list[str]:
    """Files in t3-2/logs that pack.py would hand in but that hold practice play: sandbox memory / reveal files,
    files from an earlier day, and a decisions.jsonl / console log with a sandbox game in it (practice this morning)."""
    today = time.strftime("%Y%m%d")
    out = []
    for p in sorted(hand_in.glob("*")):
        if not p.is_file():
            continue
        if "sandbox" in p.name or "reveal-" in p.name or time.strftime("%Y%m%d", time.localtime(p.stat().st_mtime)) < today:
            out.append(p.name)
            continue
        if p.suffix in (".jsonl", ".txt", ".log"):
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    if any('"market": "sandbox"' in line or "market=sandbox" in line for line in f):
                        out.append(p.name)
            except OSError:
                pass
    return out


def run_checks() -> dict:
    env = read_env()
    url = (os.environ.get("MARKET_URL") or env.get("MARKET_URL") or "").rstrip("/")
    key = os.environ.get("TEAM_KEY") or env.get("TEAM_KEY")
    res = []

    def add(name, level, detail):
        res.append({"name": name, "level": level, "detail": detail})

    port = urlparse(url).port
    comp = is_competition(url)
    url_row = len(res)
    add("ที่อยู่ตลาด", "ok" if url.startswith("http") and "/trade" not in url else "bad",
        f"{url or '(ว่าง)'}" + (" · ตลาดแข่ง" if comp else " · ตลาดฝึก" if port == 8000 else " · ตลาดในเครื่อง"))
    code, ver, ms = _get(url + "/v1/version")
    mode = ver.get("mode") if isinstance(ver, dict) else None
    add("เซิร์ฟเวอร์ตลาด", "ok" if code == 200 else "bad",
        f"ตอบใน {ms} ms · โหมด {mode}" if code == 200 else f"ติดต่อไม่ได้: {code or ver}")
    if not key:
        add("key ของทีม", "bad", "ยังไม่ได้ใส่ TEAM_KEY")
    else:
        code, prof, ms = _get(url + "/v1/profile", key)
        add("key ของทีม", "ok" if code == 200 else "bad",
            f"ใช้ได้ · ทีม {prof.get('team_id')} ({prof.get('shown_as')})" if code == 200 and isinstance(prof, dict)
            else f"ไม่ผ่าน ({code}) ตรวจว่าใช้ key ถูกตลาด: แข่ง = Key-test, ฝึก = Key-train")
        code, st, ms = _get(url + "/v1/state", key)
        if code == 200 and isinstance(st, dict):
            if not comp and is_competition(url, st.get("market")):   # scored market on another port
                comp = True
                res[url_row]["detail"] = f"{url} · ตลาดแข่ง (ตลาดแจ้ง market={st.get('market')})"
            rnd = (st.get("round") or {}).get("round_id")
            add("สถานะตลาด", "ok", f"phase {st.get('phase')}" + (f" · รอบ {rnd}" if rnd else "") +
                (f" · เหลือ {round(st['seconds_left'])} วิ" if st.get("seconds_left") is not None else "") +
                (" · หยุดชั่วคราว" if st.get("paused") else "") + f" · ล็อต {len(st.get('lots') or [])}")
        else:
            add("สถานะตลาด", "bad", f"อ่านไม่ได้ ({code})")
        code, cal, ms = _get(url + "/v1/calendar", key)
        if code == 200 and isinstance(cal, dict):
            tl = (cal.get("rules") or {}).get("timeline") or {}
            add("ปฏิทินและกติกา", "ok", f"ซ้อม {tl.get('practice_rounds')} + จริง {tl.get('rounds')} รอบ · เปิดรับ "
                f"{tl.get('open_s')} วิ · capacity factor {(cal.get('rules') or {}).get('round_capacity_factor')} · "
                f"ปฏิทิน {len(cal.get('calendar') or [])} ช่วง")
        else:
            add("ปฏิทินและกติกา", "bad", f"อ่านไม่ได้ ({code})")
    # agent settings for the day
    s = {k: (os.environ.get(k) or env.get(k) or "") for k in ("STRATEGY", "TACTIC", "USE_LLM", "PATTERN", "SPEC_P", "FREE_TRY")}
    good = s["STRATEGY"] == "adaptive" and s["TACTIC"] == "1" and s["PATTERN"] in ("", "0")
    add("ค่าตั้ง agent", "ok" if good else "warn",
        f"STRATEGY={s['STRATEGY'] or '(ว่าง)'} TACTIC={s['TACTIC'] or '0'} USE_LLM={s['USE_LLM'] or '1'} "
        f"PATTERN={s['PATTERN'] or '0'} SPEC_P={s['SPEC_P'] or '0.5'} FREE_TRY={s['FREE_TRY'] or '1'}"
        + ("" if good else " · ค่าที่ทดสอบแล้ว: STRATEGY=adaptive, TACTIC=1, PATTERN=0"))
    # LLM rule (event gateway only in a scored stage)
    if (os.environ.get("USE_LLM") or env.get("USE_LLM") or "1") == "1":
        sys.path.insert(0, str(HERE))
        from main import llm_allowed
        ok, why = llm_allowed(url, mode, os.environ.get("LLM_BASE_URL") or env.get("LLM_BASE_URL", ""),
                              os.environ.get("LLM_MODEL") or env.get("LLM_MODEL", ""))
        add("กติกา LLM", "ok" if ok else "warn", why + ("" if ok else " (agent จะใช้กฎอย่างเดียวเอง)"))
    else:
        add("กติกา LLM", "ok", "USE_LLM=0 ใช้กฎอย่างเดียว ไม่เรียก LLM")
    # logs where pack.py looks for them
    log_dir = Path(os.environ["LOG_DIR"]) if os.environ.get("LOG_DIR") else HERE / "logs"
    in_pack = HERE in log_dir.resolve().parents or log_dir.resolve() == (HERE / "logs").resolve()
    dec = log_dir / "decisions.jsonl"
    add("ที่เก็บ log", "ok" if in_pack else ("bad" if comp else "warn"),
        f"{log_dir} · decisions.jsonl {dec.stat().st_size // 1024 if dec.exists() else 0} KB · log ต่อท้าย ไม่เขียนทับ"
        + ("" if in_pack else " · อยู่นอกโฟลเดอร์ t3-2: pack.py จะไม่ส่ง log นี้"))
    hand_in = HERE / "logs"                                  # what pack.py zips for the judges
    old = practice_in_hand_in(hand_in)
    add("log ที่จะส่งงาน", "ok" if not old else ("bad" if comp else "warn"),
        "t3-2/logs ไม่มี log ตลาดฝึก" if not old else
        f"t3-2/logs มี log ตลาดฝึก/วันก่อน {len(old)} ไฟล์ (เช่น {old[0]}): ย้ายออกก่อนแข่ง pack.py ส่งทั้งโฟลเดอร์")
    others = other_agents()
    add("agent ซ้ำ", "bad" if others else "ok",
        f"มี main.py อื่นรันอยู่ pid {others}: สองตัว = ส่งซ้ำ ต้องปิดก่อน" if others else "ไม่มี agent ตัวอื่นรันอยู่")
    return {"checks": res, "competition": comp, "time": time.strftime("%H:%M:%S")}


# ------------------------------------------------------------------ agent process
def running() -> bool:
    p = _PROC["proc"]
    return bool(p and p.poll() is None)


def _reader(proc) -> None:
    for raw in iter(proc.stdout.readline, b""):
        with _LOCK:
            _PROC["seq"] += 1
            _PROC["lines"].append({"seq": _PROC["seq"], "text": raw.decode("utf-8", "replace").rstrip()})


def agent_start(forever: bool) -> dict:
    if running():
        return {"ok": False, "error": f"agent รันอยู่แล้ว (pid {_PROC['proc'].pid})"}
    others = other_agents()
    if others:
        return {"ok": False, "error": f"มี main.py อื่นรันอยู่ pid {others}: ปิดตัวนั้นก่อน (สองตัว = ส่งซ้ำ)"}
    args = [sys.executable, "-u", str(HERE / "main.py")] + (["--forever"] if forever else [])
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(args, cwd=HERE.parent, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            creationflags=flags)
    _PROC.update(proc=proc, started=time.time(), args=" ".join(a for a in args[1:]))
    with _LOCK:
        _PROC["seq"] += 1
        _PROC["lines"].append({"seq": _PROC["seq"], "text": f"--- เริ่ม agent {time.strftime('%H:%M:%S')} "
                                                            f"{'--forever' if forever else ''}"})
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    return {"ok": True, "pid": proc.pid}


def agent_stop() -> dict:
    if not running():
        return {"ok": False, "error": "agent ไม่ได้รันจากหน้านี้"}
    _PROC["proc"].terminate()
    try:
        _PROC["proc"].wait(8)
    except subprocess.TimeoutExpired:
        _PROC["proc"].kill()
    with _LOCK:
        _PROC["seq"] += 1
        _PROC["lines"].append({"seq": _PROC["seq"], "text": f"--- หยุด agent {time.strftime('%H:%M:%S')}"})
    return {"ok": True}


def agent_status(after: int = 0) -> dict:
    p = _PROC["proc"]
    with _LOCK:
        lines = [x for x in _PROC["lines"] if x["seq"] > after]
    return {"running": running(), "pid": p.pid if p else None, "exit": p.poll() if p else None,
            "started": _PROC["started"], "seq": _PROC["seq"], "lines": lines[-200:]}


# ------------------------------------------------------------------ hand-in
def pack(send: bool) -> dict:
    args = [sys.executable, str(HERE / "pack.py")] + (["--send"] if send else [])
    try:
        out = subprocess.run(args, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=120, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        text = (out.stdout or "") + (out.stderr or "")
        for k in SECRETS:                                    # never echo a key back to the page
            v = os.environ.get(k)
            if v:
                text = text.replace(v, "****")
        return {"ok": out.returncode == 0, "output": text[-3000:]}
    except Exception as e:                                   # noqa: BLE001
        return {"ok": False, "output": repr(e)}
