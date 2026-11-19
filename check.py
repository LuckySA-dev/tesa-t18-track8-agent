"""Check every market endpoint and the LLM from t3-2/.env before letting the agent play.

    python t3-2/check.py            # read-only: every GET endpoint + LLM models / chat / JSON (changes nothing)
    python t3-2/check.py --write    # sandbox only: also reset, appraise, offer, edit, cancel, batch (starts a new game!)

Never run --write on the competition market: sending offers by hand there is disqualification.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from appraisal import appraisal_items, estimate_lots   # noqa: E402
from llm import LLM                                    # noqa: E402
from main import llm_allowed, load_env                 # noqa: E402
from market import Market, MarketError                 # noqa: E402


def step(name, fn):
    try:
        out = fn()
        text = json.dumps(out, ensure_ascii=False, default=str)
        print(f"OK   {name:<24} {text[:150]}")
        return out
    except MarketError as e:
        print(f"ERR  {name:<24} {e.status} {e}")
    except Exception as e:                       # noqa: BLE001 - a check reports, never crashes
        print(f"FAIL {name:<24} {type(e).__name__}: {str(e)[:150]}")
    return None


def wait_phase(api, phases, limit=400):
    import time
    for _ in range(limit):
        s = api.state()
        if s["phase"] in phases:
            return s
        time.sleep(min(5, max(0.5, (s.get("seconds_left") or 1) + 0.2)))
    return api.state()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Thai / "·" in replies crashed cp874 consoles
    load_env(HERE / ".env")
    api = Market(os.environ["MARKET_URL"], os.environ["TEAM_KEY"])
    print(f"market {os.environ['MARKET_URL']}")
    v = step("GET /v1/version", api.version)
    s = step("GET /v1/state", api.state)
    for name, fn in [("GET /v1/calendar", api.calendar), ("GET /v1/lots", api.lots), ("GET /v1/rfqs", api.rfqs),
                     ("GET /v1/appraisals", api.appraisals), ("GET /v1/offers", api.offers),
                     ("GET /v1/results", api.results), ("GET /v1/standings", api.standings),
                     ("GET /v1/leaderboard", api.leaderboard), ("GET /v1/profile", api.profile),
                     ("GET /v1/sandbox/reveal", api.sandbox_reveal)]:
        step(name, fn)

    ok, why = llm_allowed(os.environ["MARKET_URL"], (v or {}).get("mode"), os.environ.get("LLM_BASE_URL", ""),
                          os.environ.get("LLM_MODEL", ""))
    if os.environ.get("USE_LLM", "1") == "1":
        print(f"{'OK  ' if ok else 'STOP'} LLM guard               {why}")
    if os.environ.get("USE_LLM", "1") == "1" and ok:
        llm = LLM(os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"], os.environ.get("LLM_API_KEY", ""), timeout=60,
                  max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "700")))
        print(f"llm {os.environ['LLM_BASE_URL']} model {os.environ['LLM_MODEL']}")
        step("LLM models", llm.models)
        # 2 gateway calls only (organiser limit: no more than 2 calls back to back, <= 20 / min)
        schema = {"type": "object", "required": ["k"], "properties": {"k": {"type": "number"}}}
        step("LLM ask_json", lambda: llm.ask_json("Answer only JSON.", "Return k = 1.0", schema) or llm.last_error)

    if "--write" not in sys.argv:
        return
    if not v or v.get("mode") != "sandbox":
        sys.exit("--write is for the sandbox only (version.mode is not 'sandbox'): stopped")
    step("POST /v1/sandbox/reset", api.sandbox_reset)
    s = wait_phase(api, {"APPRAISAL"})
    rules = api.calendar()["rules"]
    tv = estimate_lots(rules, s["lots"])
    step("POST /v1/appraisals", lambda: api.appraise(appraisal_items(rules, s["lots"], tv)))
    step("POST appraisals again", lambda: api.appraise(appraisal_items(rules, s["lots"], tv)))   # expect 409
    s = wait_phase(api, {"ROUND_OPEN"})
    lot, rfq = s["lots"][0], s["rfqs"][0]
    o = step("POST /v1/offers", lambda: api.offer({"rfq_id": rfq["rfq_id"], "lot_id": lot["lot_id"], "kg": 60,
                                                    "price_per_kg": round(1.2 * tv[lot["lot_id"]], 2)}))
    step("POST offers 61 kg", lambda: api.offer({"rfq_id": rfq["rfq_id"], "lot_id": lot["lot_id"], "kg": 61,
                                                  "price_per_kg": 100}))                     # expect NOT_MULTIPLE_OF_BAG
    if o:
        step("PUT /v1/offers/{id}", lambda: api.update_offer(o["offer_id"], price_per_kg=round(1.1 * tv[lot["lot_id"]], 2)))
        step("DELETE /v1/offers/{id}", lambda: api.cancel_offer(o["offer_id"]))
    step("POST /v1/offers:batch", lambda: api.offer_batch([{"rfq_id": rfq["rfq_id"], "lot_id": lot["lot_id"], "kg": 60,
                                                            "price_per_kg": round(1.3 * tv[lot["lot_id"]], 2)}]))
    step("GET /v1/offers", api.offers)
    rid = s["round"]["round_id"]
    wait_phase(api, {"ROUND_CLEARING", "CLOSED"})          # this round closes ...
    wait_phase(api, {"ROUND_OPEN", "CLOSED"})              # ... and is cleared
    step(f"GET /v1/results?round={rid}", lambda: api.results(rid))


if __name__ == "__main__":
    main()
