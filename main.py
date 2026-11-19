"""Coffee market starter agent: poll the market, appraise once, send offers every round.

    python t3-2/main.py             # play until the market closes
    python t3-2/main.py --forever   # sandbox: keep playing game after game
"""
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import strategy
from appraisal import appraisal_items, estimate_lots
from llm import LLM
from market import Market, MarketError
from memory import Memory

HERE = Path(__file__).parent
LOGS = Path(os.environ["LOG_DIR"]) if os.environ.get("LOG_DIR") else HERE / "logs"   # LOG_DIR: mock / test runs

# The only LLM allowed in a scored stage (endpoint.txt, slides 18 Nov p.27-28). Any other LLM there = disqualified.
GATEWAY_HOST = "202.29.227.22:4000"
GATEWAY_MODEL = "event-llm"


def llm_allowed(market_url: str, market_mode: str | None, base_url: str, model: str) -> tuple[bool, str]:
    """Practice sandbox: any LLM. Anything else (competition port 8001, mode not 'sandbox', or mode unknown because
    /v1/version failed): only the event gateway, otherwise the agent runs on rules alone."""
    host = urlparse(base_url or "").netloc
    if ":8001" not in market_url and market_mode == "sandbox":
        return True, "practice sandbox"
    if host == GATEWAY_HOST and model == GATEWAY_MODEL:
        return True, "event gateway"
    return False, (f"scored market (url {urlparse(market_url).netloc}, mode {market_mode}): LLM '{model}' at "
                   f"'{host}' is not the event gateway {GATEWAY_HOST} / {GATEWAY_MODEL} -> rules only")


def load_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def log(event: str, **data) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **data}
    with open(LOGS / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    short = {k: v for k, v in data.items() if k not in ("offers", "notes", "tactic")}
    print(row["time"][11:], event, json.dumps(short, ensure_ascii=False)[:160], flush=True)


SESSION = LOGS / f"session-{datetime.now():%Y%m%d-%H%M%S}.md"   # one human-readable file per agent run


def session_round(state: dict, memory: Memory, tv: dict, decided: dict) -> None:
    """After every cleared round: one event in decisions.jsonl + one block in the session log (never raises)."""
    try:
        rnd = memory.rounds[-1]
        rid = rnd["round_id"]
        mine = rnd.get("my_offers", [])
        meta = decided.get(rid, {})
        sold = sum(o.get("filled_kg", 0) for o in mine)
        rev = sum(o.get("filled_kg", 0) * o["price_per_kg"] for o in mine)
        k_sold = sum(o.get("filled_kg", 0) * o["price_per_kg"] / tv[o["lot_id"]] for o in mine if o["lot_id"] in tv)
        st = state.get("standings") or {}
        total = sum(tv.get(l["lot_id"], 0) * l["kg_total"] for l in state["lots"])
        left = sum(tv.get(l["lot_id"], 0) * l["kg_remaining"] for l in state["lots"])
        pr_now = (st.get("revenue_thb", 0) + 0.8 * left) / total if total else None
        statuses = {}
        for o in mine:
            statuses[o["status"]] = statuses.get(o["status"], 0) + 1
        log("round_result", round=rid, kind=rnd.get("kind"), offered_kg=sum(o["kg"] for o in mine), sold_kg=sold,
            revenue=round(rev, 2), k_sold=round(k_sold / sold, 4) if sold else None, statuses=statuses,
            source=meta.get("source"), seconds=meta.get("seconds"), pr_now=round(pr_now, 4) if pr_now else None)
        lines = [f"\n## {rid} ({rnd.get('kind', 'real')}) · cleared {datetime.now():%H:%M:%S}",
                 f"- decided by **{meta.get('source', '?')}** in {meta.get('seconds', '?')} s"
                 + (f" · LLM: {meta['llm']}" if meta.get("llm") else ""),
                 f"- offered {sum(o['kg'] for o in mine)} kg in {len(mine)} offers · sold {sold} kg · revenue {rev:,.0f} THB"
                 + (f" · avg k sold {k_sold / sold:.3f}" if sold else ""),
                 "- statuses: " + (", ".join(f"{s} {n}" for s, n in sorted(statuses.items())) or "none"),
                 f"- standings: sold {st.get('kg_sold')} / {st.get('kg_total')} kg · revenue {st.get('revenue_thb')} THB"
                 + (f" · PR now {pr_now:.4f} (rest salvaged)" if pr_now else ""),
                 "", "| RFQ | lot | kg | price | k | status | filled |", "|---|---|---|---|---|---|---|"]
        for o in mine:
            k = o["price_per_kg"] / tv[o["lot_id"]] if o["lot_id"] in tv else 0
            lines.append(f"| {o['rfq_id']} | {o['lot_id']} | {o['kg']} | {o['price_per_kg']} | {k:.3f} | {o['status']} | "
                         f"{o.get('filled_kg', 0)} |")
        lines += ["", "| RFQ | type | wanted | bought | offers | min | avg | max |", "|---|---|---|---|---|---|---|---|"]
        for m in rnd.get("market_summary", []):
            lines.append(f"| {m['rfq_id']} | {m['buyer_type']} | {m['wanted_kg']} | {m['bought_kg']} | {m['offers']} | "
                         f"{m.get('price_min')} | {m.get('price_avg')} | {m.get('price_max')} |")
        LOGS.mkdir(parents=True, exist_ok=True)
        new = not SESSION.exists()
        with open(SESSION, "a", encoding="utf-8") as f:
            if new:
                f.write(f"# Session log {datetime.now():%Y-%m-%d %H:%M} · market {state.get('market')} · "
                        f"LLM {os.environ.get('LLM_MODEL') if os.environ.get('USE_LLM', '1') == '1' else 'off'}\n")
            f.write("\n".join(lines) + "\n")
    except Exception as e:                              # noqa: BLE001 - logging must never stop the agent
        print("session log failed:", repr(e), flush=True)


def close_report(api: Market, state: dict, memory: Memory, tv: dict, lots_key) -> None:
    """At CLOSED: fill rounds the agent missed (restart) from GET /v1/results, and in the sandbox save the reveal
    (true quality + score) for research/calibrate_live.py. Read-only calls; never raises."""
    try:
        for rid in list(memory.sent):
            if not any(r["round_id"] == rid for r in memory.rounds):
                try:
                    memory.record_results(api.results(rid))
                except MarketError:
                    pass
        if state.get("market") != "sandbox":
            return
        rev = api.sandbox_reveal()
        tag = hashlib.sha1(repr(lots_key).encode()).hexdigest()[:8]
        path = LOGS / f"reveal-{datetime.now():%Y%m%d-%H%M}-{tag}.json"
        path.write_text(json.dumps(rev, ensure_ascii=False, indent=1), encoding="utf-8")
        sc = rev.get("score") or {}
        lines = [f"\n## Game closed {datetime.now():%H:%M:%S} · reveal saved to {path.name}",
                 f"- PR {sc.get('pr')} · MAPE {sc.get('mape')} · appraisal {sc.get('appraisal_score')} · "
                 f"sell {sc.get('pr_score')} · total {sc.get('total')}",
                 "", "| lot | sample d / m / s | true d / m / s | noise d | noise m | true TV | our TV |",
                 "|---|---|---|---|---|---|---|"]
        for l in rev.get("lots", []):
            s, t = l["sample"], l["true"]
            lines.append(f"| {l['lot_id']} | {s['defect_pct']} / {s['moisture_pct']} / {s['screen']} | "
                         f"{t['defect_pct']} / {t['moisture_pct']} / {t['screen']} | "
                         f"{s['defect_pct'] - t['defect_pct']:+.2f} | {s['moisture_pct'] - t['moisture_pct']:+.2f} | "
                         f"{l.get('true_value_per_kg')} | {tv.get(l['lot_id'])} |")
        with open(SESSION, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        log("reveal_saved", file=path.name, pr=sc.get("pr"), mape=sc.get("mape"))
    except Exception as e:                              # noqa: BLE001
        print("close report failed:", repr(e), flush=True)


def send_offers(api: Market, offers: list[dict]) -> list[dict]:
    """Send all offers at once; if the batch is rejected, send them one by one and skip the bad ones."""
    try:
        return api.offer_batch(offers)["offers"]
    except MarketError as e:
        log("batch_rejected", error=str(e))
    sent = []
    for o in offers:
        try:
            sent.append(api.offer(o))
        except MarketError as e:
            log("offer_rejected", offer=o, error=str(e))
    return sent


def round_order(rid: str | None) -> int:
    """P1 < P2 < ... < R1 < R2 ...; None (SETUP / APPRAISAL) is before everything."""
    if not rid:
        return -1
    digits = "".join(c for c in rid if c.isdigit())   # any id shape ("R1", "PR1", "ROUND-03"): never raise
    return (0 if rid[0].upper() == "P" else 100) + (int(digits[-6:]) if digits else 0)


def stale_memory(memory: Memory, state: dict) -> bool:
    """A memory file whose rounds lie AFTER where the market is now belongs to an earlier game with the same lots
    (a restarted market with the same seed): its results would steer this game. A restart mid-game keeps it."""
    if not memory.rounds and not memory.sent:
        return False
    now = round_order((state.get("round") or {}).get("round_id")) if state["phase"] not in ("SETUP", "APPRAISAL", "WAITING") else -1
    if state["phase"] == "CLOSED":
        return False
    seen = [round_order(r.get("round_id")) for r in memory.rounds] + [round_order(k) for k in memory.sent]
    return max(seen) > now if seen else False


def play(api: Market, llm: LLM | None, forever: bool) -> None:
    info = api.calendar()
    rules = info["rules"]
    memory, tv, acted, game, n_teams = None, {}, set(), None, None
    decided = {}                                    # round_id -> how it was decided (session log)
    rng = random.Random()
    while True:
        try:
            state = api.state()
        except Exception as e:                      # noqa: BLE001 — never die on a network error
            log("state_error", error=str(e))
            time.sleep(3)
            continue
        # the sandbox reuses lot ids every game, so the samples are what tell one game from the next
        lots_key = tuple(sorted((l["lot_id"], json.dumps(l["sample"], sort_keys=True)) for l in state["lots"]))
        # same lots again (market restarted with the same seed): the market clock going back means a new game too,
        # otherwise `acted` still holds this game's round ids and every round would be skipped
        pos = -1 if state["phase"] in ("SETUP", "APPRAISAL", "WAITING") else round_order((state.get("round") or {}).get("round_id"))
        if game is not None and lots_key == game and acted and pos < max(
                [round_order(r) for r in acted if r != "closed"] or [-1]) and state["phase"] != "CLOSED":
            log("market_restarted", reason="market clock went back with the same lots")
            game = None
        if lots_key != game:                        # new game (or sandbox reset): start fresh
            try:                                    # rules / calendar can change between games (Day 17 -> Day 18)
                info = api.calendar()
                rules = info["rules"]
            except Exception as e:                  # noqa: BLE001 - keep the previous rules
                log("calendar_error", error=str(e))
            try:                                    # sellers in the room: market supply = teams x our bag count
                n_teams = len(api.leaderboard().get("ranking") or []) or None
            except Exception:                       # noqa: BLE001 - optional, position works without it
                n_teams = None
            if not n_teams and os.environ.get("N_TEAMS", "").isdigit():   # leaderboard down / empty
                n_teams = int(os.environ["N_TEAMS"]) or None
            game, acted, tv, decided = lots_key, set(), estimate_lots(rules, state["lots"]), {}
            tag = hashlib.sha1(repr(lots_key).encode()).hexdigest()[:8]
            memory = Memory(LOGS / f"memory-{state['market']}-{tag}.json")
            if stale_memory(memory, state):         # same lots again (same seed / reset): an older game's memory
                old = memory.path.with_name(memory.path.stem + f"-old-{time.strftime('%H%M%S')}.json")
                memory.path.rename(old)
                log("memory_reset", reason="memory is from an earlier game with the same lots", archived=old.name)
                memory = Memory(LOGS / f"memory-{state['market']}-{tag}.json")
            log("new_game", market=state["market"], lots=len(state["lots"]), tv=tv, n_teams=n_teams,
                capacity_factor=rules.get("round_capacity_factor"), rounds=rules["timeline"]["rounds"])
        phase = state["phase"]

        if phase == "APPRAISAL" and not state["appraisal_submitted"]:
            try:
                api.appraise(appraisal_items(rules, state["lots"], tv))
                log("appraisal_sent", tv=tv)
            except MarketError as e:
                log("appraisal_error", error=str(e))

        elif phase == "ROUND_OPEN" and not state["paused"] and state["round"]["round_id"] not in acted:
            rid = state["round"]["round_id"]
            acted.add(rid)
            if memory.record_results(state["last_results"]):
                session_round(state, memory, tv, decided)
            if any(o["round_id"] == rid for o in state["my_open_offers"]):
                # restarted after sending this round: the market already holds our offers, and a new batch
                # would replace them (capacity_left_kg no longer counts them, so it would offer less)
                log("round_already_sent", round=rid, n=len(state["my_open_offers"]))
                continue
            ctx = strategy.Context(rules, info["calendar"], tv, rng, memory, llm, state["seconds_left"] or 999,
                                   n_teams=n_teams)
            t0 = time.time()
            try:
                offers, source = strategy.decide(state, ctx)
            except Exception as e:                  # noqa: BLE001 — a bug in strategy must not stop the agent
                log("strategy_crashed", error=repr(e))
                offers, source = strategy.guard(state, ctx, strategy.rule_plan(state, ctx, strategy.candidates(state, ctx))), "rules"
            sent = send_offers(api, offers) if offers else []
            memory.record_sent(rid, offers, state, tv)
            decided[rid] = {"source": source, "seconds": round(time.time() - t0, 1),
                         "llm": next((n for n in ctx.notes if n.lower().startswith("llm")), "")[:160]}
            log("offers_sent", round=rid, source=source, n=len(sent), kg=sum(o["kg"] for o in sent),
                seconds=round(time.time() - t0, 1), offers=offers, tactic=getattr(ctx, "tactic", None), notes=ctx.notes)

        elif phase == "CLOSED" and game is not None and "closed" not in acted:
            acted.add("closed")
            if memory.record_results(state["last_results"]):
                session_round(state, memory, tv, decided)
            log("market_closed", standings=state["standings"])
            close_report(api, state, memory, tv, lots_key)
            if not forever:
                return

        # Read the clock from the server; never hard-code round times.
        left = state["seconds_left"] or 5
        time.sleep(min(5.0, max(0.5, left + 0.2)))


def main() -> None:
    load_env(HERE / ".env")
    if os.environ.get("PRIORS_FILE"):                # pricing loaded priors at import, before .env was read
        import pricing
        pricing.load_priors()
        log("priors_loaded", file=os.environ["PRIORS_FILE"], buyers=len(pricing.BUYER_PRIORS))
    api = Market(os.environ["MARKET_URL"], os.environ["TEAM_KEY"])
    llm = None
    if os.environ.get("USE_LLM", "1") == "1":
        try:
            mode = api.version().get("mode")
        except Exception:                           # noqa: BLE001 - unknown mode is treated as a scored market
            mode = None
        ok, why = llm_allowed(os.environ["MARKET_URL"], mode, os.environ.get("LLM_BASE_URL", ""),
                              os.environ.get("LLM_MODEL", ""))
        log("llm_guard", allowed=ok, reason=why)
    if os.environ.get("USE_LLM", "1") == "1" and ok:
        llm = LLM(os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"], os.environ.get("LLM_API_KEY", ""),
                  timeout=float(os.environ.get("LLM_TIMEOUT", "60")), thinking=os.environ.get("LLM_THINKING") == "1",
                  max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "700")))   # gateway context is ~4,000 tokens
    play(api, llm, "--forever" in sys.argv)


if __name__ == "__main__":
    main()
