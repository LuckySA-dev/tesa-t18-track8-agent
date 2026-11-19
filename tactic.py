"""Tactic layer (TACTIC=1): reacts to our own results before the guard sends a plan.

The planner (and the LLM editor) choose k per offer from learnt ceilings. Stage 2 (18 Nov) and the mocks showed
two opposite failures that nothing corrected round to round:
  - too high: R1-R4 of stage 2 sold 0 kg (the LLM lifted Commercial 0.97 -> 1.05: ABOVE_LIMIT, then OUTBID);
  - too low: mock20 sold everything at the 0.85 floor by R10 while rivals were paid ~1.0-1.1.
This layer is a feedback controller on top of whatever plan comes in. Every step is logged in ctx.notes.

Defaults since 19 Nov night (blind test, 700 unseen random markets, paired: PR +0.0032 [+0.0024, +0.0040], top 5
58 -> 62 %, rank 1 229 -> 269; tuned on 300 other markets): UP 0.03, DOWN_OUTBID 0.02, ATTACK +0.02 at cover < 0.7.
1. Per-buyer feedback (AIMD on k, from memory.rounds + memory.sent):
     ACCEPTED at k (filled)   -> next k for that buyer = k + UP (ask a little more: "ACCEPTED every time = too cheap")
     PARTIAL at k             -> keep k
     OUTBID at k              -> next k = min(k - DOWN_OUTBID, k_max_sold - EDGE) (k_max_sold = the highest price
                                 that RFQ paid, divided by the TV of our lot: every team has the same lot set)
     ABOVE_LIMIT at k         -> hard cap for that buyer at k - CAP_GAP for the rest of the game, next k = k - DOWN_ABOVE
   A buyer we never sold to inherits the latest signal of its buyer type.
2. Mode per round from the SELL LIMIT risk (risk()): every round places at most `cap` bags, so a round that sells
   nothing cannot be made up later. With sell-through rate r (sold / offered, recent rounds weigh more):
     RESCUE  drought >= 2 real rounds with 0 kg sold, or expected unsold > UNSOLD_RESCUE (10 %) at this pace
     SAFE    drought 1, or missing THIS round would leave stock unsold (if_miss > MISS_SAFE), or last round
     NORMAL  otherwise
     ATTACK  at least SPARE_ATTACK (3) spare rounds at this pace and the last round sold well
   (hysteresis: one step up per round; TAC_RISK=0 = the first version, cover = stock / (cap x rounds left))
   Mode shifts k: ATTACK +0.01, NORMAL 0, SAFE -0.01, RESCUE -0.03 (suite sweep 19 Nov); RESCUE/SAFE also fill the round capacity.
3. LLM clamp: a row the LLM changed may move at most LLM_DELTA from the planner's k for that pair, and never above
   the buyer's cap (stage 2 failure 1).
4. Spread: one offer takes at most SPREAD of the round capacity when another RFQ can take the rest, so one OUTBID
   does not zero the round (stage 2: both lots on one RFQ -> whole round lost).
   ABOVE_LIMIT also caps buyers of the same type we have not offered to yet. Last round: k stays under the highest
   k that buyer type accepted from us in the last 3 rounds - LAST_GAP (live mock20 R15: ABOVE_LIMIT, 4 bags left).
6. Market anchor (TAC_ANCHOR=1): k never above what that buyer paid lately + PREMIUM[mode] (ATTACK +0.01,
   NORMAL -0.005, SAFE -0.02, RESCUE -0.04). Paid = average price of a filled RFQ (competition sets it), top price of an
   under-filled RFQ (its ceiling binds), over the TV of our lots that pass its spec. Stage 2: priors said Commercial
   ceiling 1.06, rivals were paid 0.97-1.00 -> the anchor keeps us at the rivals' level instead of the ceiling.
   Off by default: on the real stage-2 rounds it read Commercial at k 1.03-1.07 (rivals sold another lot mix; they
   were paid ~0.97-1.00), and in the replica it cost PR in 10 of 11 markets. TAC_OUTBID_REF=avg (aim OUTBID at the
   average price) also lost PR; default max. The last round goes RESCUE when the stock is
   more than the round can place (TAC_LAST_RESCUE=1).
7. Lot TV correction (TAC_LOT_TV=1, off: suite 19 Nov no gain, mock20 seed 11 overshot to 0.75): a lot refused (ABOVE_LIMIT) while other lots sold to that buyer type at the
   same or higher k has a too-high TV estimate; each such refusal cuts that lot's k by LOT_CUT (4 %), floor included.
8. Win-rate pricing (TAC_WINRATE=1, off unless proven): per buyer type, P(win | k) from our last 6 rounds of offers
   (sale at k' >= k -> k sells; loss at k' <= k -> k loses; logistic prior around the tactic k), then
   k = argmax (k - keep) x P(k) on a 0.005 grid from k - 0.06 to k + 0.04. keep = what the bag is worth if not sold
   now: 0.80 when behind or in the last round, else our recent sold k - WR_KEEP_GAP. TAC SCM sales agents and
   first-price bid shading work this way (estimate win probability by price, maximise expected profit).
9. Reserve (TAC_RESERVE=1, off unless proven): when rivals' stock left / demand left < RES_PRESSURE (0.9) and our cover
   < RES_COVER (0.7) with 3+ rounds left, offer only an even share of our stock this round and ask RES_PREMIUM more
   (literature: rivals' scarcity lifts later prices; Budde-Minner: capacity-limited sellers bid higher when repeated).
5. Floor: k never below salvage + FLOOR_GAP (0.80 + 0.02); the guard's 0.85 floor moves down only in RESCUE and in
   the last round, where anything above 0.80 beats keeping the bag.
"""
from __future__ import annotations

import os
from collections import defaultdict

BAG = 60
SALVAGE = 0.80


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def params() -> dict:
    return {"up": _f("TAC_UP", 0.03), "down_outbid": _f("TAC_DOWN_OUTBID", 0.02), "down_above": _f("TAC_DOWN_ABOVE", 0.04),
            "cap_gap": _f("TAC_CAP_GAP", 0.015), "edge": _f("TAC_EDGE", 0.005), "llm_delta": _f("TAC_LLM_DELTA", 0.02),
            "spread": _f("TAC_SPREAD", 0.6), "floor_gap": _f("TAC_FLOOR_GAP", 0.02),
            "cover_safe": _f("TAC_COVER_SAFE", 0.85), "cover_attack": _f("TAC_COVER_ATTACK", 0.7),
            "rate_prior": _f("TAC_RATE_PRIOR", 0.7), "unsold_rescue": _f("TAC_UNSOLD_RESCUE", 0.10),
            "miss_safe": _f("TAC_MISS_SAFE", 0.0), "spare_attack": _f("TAC_SPARE_ATTACK", 3.0),
            "unsold_override": _f("TAC_UNSOLD_OVERRIDE", 0.35), "last_gap": _f("TAC_LAST_GAP", 0.01), "lot_cut": _f("TAC_LOT_CUT", 0.04), "max_drop": _f("TAC_MAX_DROP", 0.08), "wr_keep_gap": _f("TAC_WR_KEEP_GAP", 0.04),
            "res_pressure": _f("TAC_RES_PRESSURE", 0.9), "res_cover": _f("TAC_RES_COVER", 0.7),
            "res_premium": _f("TAC_RES_PREMIUM", 0.02), "miss_override": _f("TAC_MISS_OVERRIDE", 0.35),
            "premium": {"ATTACK": _f("TAC_PREM_ATTACK", 0.01), "NORMAL": -_f("TAC_PREM_NORMAL", 0.005),
                        "SAFE": -_f("TAC_PREM_SAFE", 0.02), "RESCUE": -_f("TAC_PREM_RESCUE", 0.04)},
            "shift": {"ATTACK": _f("TAC_SHIFT_ATTACK", 0.02), "NORMAL": 0.0, "SAFE": -_f("TAC_SHIFT_SAFE", 0.01),
                      "RESCUE": -_f("TAC_SHIFT_RESCUE", 0.03)}}


def _type(t: str) -> str:
    t = (t or "").upper()
    return "SPECIALTY" if t.startswith("SPE") else "COMMERCIAL" if t.startswith("COM") else "INSTANT" if t.startswith("INS") else t


def feedback(memory, tv: dict, p: dict) -> tuple[dict, dict, dict, list, dict]:
    """Walk our past offers oldest first. Returns (next_k by buyer, next_k by type, cap by buyer, history rows,
    extra = {"type_cap": ABOVE_LIMIT cap per buyer type, "type_acc": [(round index, k)] accepted per type})."""
    nxt, nxt_type, cap = {}, {}, {}
    type_cap, type_acc = {}, defaultdict(list)
    lot_acc, lot_above = defaultdict(list), []      # (idx, k, lot) per buyer type; ABOVE events (idx, lot, type, k)
    hist = []                                  # per round: {"round", "real", "offered", "sold"}
    for idx, rnd in enumerate(memory.rounds if memory else []):
        rid = rnd.get("round_id", "")
        summary = {r["rfq_id"]: r for r in rnd.get("market_summary", [])}
        sent = {(o["rfq_id"], o["lot_id"]): o for o in (memory.sent.get(rid, []) if memory else [])}
        offered = sold = 0
        for o in rnd.get("my_offers", []):
            row = summary.get(o["rfq_id"], {})
            buyer, btype = row.get("buyer", "?"), _type(row.get("buyer_type") or sent.get((o["rfq_id"], o["lot_id"]), {}).get("buyer_type", ""))
            t = tv.get(o["lot_id"])
            if not t:
                continue
            k = o["price_per_kg"] / t
            offered += o.get("kg", 0)
            sold += o.get("filled_kg", 0) or 0
            st = o.get("status")
            if st in ("ACCEPTED", "PARTIAL"):
                type_acc[btype].append((idx, k))
                lot_acc[btype].append((idx, k, o["lot_id"]))
            if st == "ABOVE_LIMIT":
                lot_above.append((idx, o["lot_id"], btype, k))
            if st == "ACCEPTED":
                n = k + p["up"]
            elif st == "PARTIAL":
                n = k
            elif st == "OUTBID":
                n = k - p["down_outbid"]
                ref = row.get("price_avg") if os.environ.get("TAC_OUTBID_REF", "max") == "avg" else None
                ref = ref or row.get("price_max")
                if ref:
                    # the THB price came from whatever lot won (another lot = another TV), so ref / our TV can read
                    # far too low (random market seed 7011: Specialty 1.18 -> 0.94): one OUTBID drops at most MAX_DROP (0.08: suites 19 Nov)
                    n = max(min(n, ref / t - p["edge"]), k - p["max_drop"])
            elif st == "ABOVE_LIMIT":
                cap[buyer] = min(cap.get(buyer, 9.0), k - p["cap_gap"])
                type_cap[btype] = min(type_cap.get(btype, 9.0), k - p["cap_gap"])
                n = k - p["down_above"]
            else:                               # SPEC_MISMATCH etc.: no price signal
                continue
            if buyer in cap:
                n = min(n, cap[buyer])
            # several offers to one buyer in a round: the most cautious signal wins
            nxt[buyer] = min(n, nxt[buyer]) if nxt.get("_r" + buyer) == rid else n
            nxt["_r" + buyer] = rid
            nxt_type[btype] = min(n, nxt_type[btype]) if nxt_type.get("_r" + btype) == rid else n
            nxt_type["_r" + btype] = rid
        hist.append({"round": rid, "real": rid.startswith("R"), "offered": offered, "sold": sold})
    clean = lambda d: {k: v for k, v in d.items() if not k.startswith("_r")}   # noqa: E731
    # 7. lot TV correction: a lot refused (ABOVE_LIMIT) at k while OTHER lots sold to the same buyer type at >= k
    #    within 3 rounds -> the buyer's ceiling did not bind, our TV for that lot is too high (sample near a grade step;
    #    mock20 seed 11: lot 1 sample defect 6.7, true 7.07, TV -13 %, ABOVE seven rounds running at the 0.85 floor).
    #    Every such refusal cuts that lot's k by LOT_CUT (4 %).
    lot_factor = {}
    for idx, lot, btype, k in lot_above:
        if any(kk >= k - 0.005 and l2 != lot and abs(i2 - idx) <= 3 for i2, kk, l2 in lot_acc.get(btype, [])):
            lot_factor[lot] = lot_factor.get(lot, 1.0) * (1 - p["lot_cut"])
    # a later sale of that lot means the correction is enough: stop cutting further (keep what was learnt)
    return clean(nxt), clean(nxt_type), cap, hist, {"type_cap": type_cap, "type_acc": dict(type_acc),
                                                   "lot_factor": lot_factor,
                                                   "rounds": len(memory.rounds) if memory else 0}


def risk(stock: int, cap: int, rounds_left: int, hist: list, prior: float) -> dict:
    """Sell-limit risk: each round can place at most `cap` bags, so a round that sells nothing is lost for good.

    rate       sell-through = sold / offered, recent real rounds weighted more (EWMA 0.5), prior until we know
    per_round  bags we expect to place per round = cap x rate
    need       rounds needed at that pace to sell everything = stock / per_round
    unsold     expected share of stock left at the end (salvage 0.80) if the pace holds
    if_miss    the same if THIS round sells nothing (one round fewer)
    stuck_at   the real round left count at which we fall behind for good (0 = never at this pace)"""
    rate = prior
    for h in hist:
        if h["real"] and h["offered"]:
            rate = 0.5 * rate + 0.5 * min(1.0, h["sold"] / h["offered"])
    per_round = max(0.25, cap * rate)
    need = stock / per_round
    unsold = max(0.0, stock - per_round * rounds_left) / max(1, stock)
    if_miss = max(0.0, stock - per_round * max(0, rounds_left - 1)) / max(1, stock)
    return {"rate": round(rate, 2), "per_round": round(per_round, 2), "need": round(need, 1),
            "spare": round(rounds_left - need, 1), "unsold": round(unsold, 3), "if_miss": round(if_miss, 3)}


def anchors(memory, lots: list[dict], tv: dict, p: dict, last: int = 2) -> dict:
    """k each buyer paid lately (organiser data in market_summary, all teams), per buyer name:
      filled RFQ (bought >= 90 % of wanted)  -> competition sets the price: anchor = price_avg / TV
      under-filled RFQ                       -> the buyer's ceiling binds: anchor = price_max / TV (it took that)
    TV = mean TV of OUR lots that pass that buyer's spec (every team gets the same lots; only ids and order differ).
    A buyer with no sale in the window has no anchor."""
    import market_core as mc
    specs = getattr(memory, "buyers", {}) or {}
    out = {}
    for rnd in (memory.rounds if memory else [])[-last:]:
        for row in rnd.get("market_summary", []):
            buyer = row.get("buyer")
            spec = (specs.get(buyer) or {}).get("spec")
            if not row.get("bought_kg") or not spec:
                continue
            ref = [tv[l["lot_id"]] for l in lots if l["lot_id"] in tv and mc.passes(l["sample"], spec)]
            if not ref:
                continue
            ref_tv = sum(ref) / len(ref)
            filled = row["bought_kg"] >= 0.9 * row["wanted_kg"]
            price = row.get("price_avg") if filled else (row.get("price_max") or row.get("price_avg"))
            if not price:
                continue
            out.setdefault(buyer, []).append((price / ref_tv, filled))
    return {b: {"k": round(sum(k for k, _ in v) / len(v), 4), "filled": all(f for _, f in v)} for b, v in out.items()}


def win_evidence(memory, tv: dict, last: int = 6) -> dict:
    """(k, won) of our offers per buyer type over the last rounds (censored feedback, like TAC SCM / bid shading)."""
    out = defaultdict(list)
    for rnd in (memory.rounds if memory else [])[-last:]:
        kinds = {m["rfq_id"]: _type(m["buyer_type"]) for m in rnd.get("market_summary", [])}
        for o in rnd.get("my_offers", []):
            t, v = kinds.get(o["rfq_id"]), tv.get(o["lot_id"])
            if not t or not v or o.get("status") not in ("ACCEPTED", "PARTIAL", "OUTBID", "ABOVE_LIMIT"):
                continue
            out[t].append((o["price_per_kg"] / v, o["status"] in ("ACCEPTED", "PARTIAL")))
    return out


def win_prob(k: float, ev: list, k_prior: float, alpha: float = 2.0, width: float = 0.025) -> float:
    """Monotone win-rate estimate: a sale at k' >= k says k sells too, a loss at k' <= k says k loses too; a logistic
    prior centred on the current tactic k fills the gaps (alpha = its weight in pseudo-offers)."""
    import math
    wins = sum(1 for kk, w in ev if w and kk >= k - 1e-9)
    loss = sum(1 for kk, w in ev if not w and kk <= k + 1e-9)
    prior = 1 / (1 + math.exp((k - k_prior) / width))
    return (wins + alpha * prior) / (wins + loss + alpha)


def rival_pressure(memory, n_teams, our_total_bags: int, rounds_left: int, unit: int = 60) -> float | None:
    """Rivals' stock left / demand left. Every team gets the same lot set, so the room's supply is n_teams x our bags;
    minus everything buyers bought in the real rounds. Demand left = average wanted per real round x rounds left.
    < 1 = buyers will be short of sellers late in the game (Hortacsu-Ory-Williams: rivals' scarcity lifts later prices)."""
    if not n_teams or not memory or not memory.rounds:
        return None
    real = [r for r in memory.rounds if str(r.get("round_id", "")).startswith("R")]
    if not real:
        return None
    bought = sum(m.get("bought_kg", 0) for r in real for m in r.get("market_summary", [])) / unit
    wanted = [sum(m.get("wanted_kg", 0) for m in r.get("market_summary", [])) / unit for r in real]
    supply_left = n_teams * our_total_bags - bought
    demand_left = (sum(wanted) / len(wanted)) * max(1, rounds_left)
    return round(supply_left / demand_left, 3) if demand_left > 0 else None


def sellable_lot_ids(state: dict, memory) -> set:
    """Lots that some known buyer might take (sample passes a spec with >= 10 % chance). Lots no buyer can take
    (defect > 15, moisture > 15: REJECT / hard limit) are left out of the sell-limit maths: every team keeps them
    at 0.80 x TV, and counting them made tactic panic (mock20 seed 0, 19 Nov: 17 REJECT bags -> SAFE, lower k)."""
    import market_core as mc
    specs = [r["spec"] for r in state.get("rfqs", [])]
    specs += [b.get("spec") for b in (getattr(memory, "buyers", None) or {}).values() if b.get("spec")]
    if not specs:
        return {l["lot_id"] for l in state["lots"]}
    noise = {"defect": 0.35, "moisture": 0.30}
    return {l["lot_id"] for l in state["lots"]
            if any(mc.pass_probability(l["sample"], sp, noise) >= 0.1 for sp in specs)}


def choose_mode(state: dict, hist: list, rounds_left: int, p: dict, prev: str | None, sellable: set | None = None) -> tuple[str, dict]:
    stock = sum(l["kg_available"] for l in state["lots"] if sellable is None or l["lot_id"] in sellable) // BAG
    cap_kg = state.get("round_capacity_kg") or state.get("capacity_left_kg")
    cap = (cap_kg or 10**9) // BAG
    cover = stock / max(1, cap * max(1, rounds_left))
    real = [h for h in hist if h["real"] and h["offered"]]
    drought = 0
    for h in reversed(real):
        if h["sold"]:
            break
        drought += 1
    last_ok = bool(real) and real[-1]["sold"] >= 0.6 * real[-1]["offered"]
    r = risk(stock, cap, rounds_left, hist, p["rate_prior"])
    risk_mode = os.environ.get("TAC_RISK", "0")      # suite 19 Nov: 0 best; 1 and 2 lower PR
    if risk_mode == "1":
        # the sell limit decides: if missing this round already leaves stock unsold, this round must sell
        if drought >= 2 or r["unsold"] > p["unsold_rescue"]:
            mode = "RESCUE"
        elif drought == 1 or r["if_miss"] > p["miss_safe"] or rounds_left <= 1:
            mode = "SAFE"
        elif r["spare"] >= p["spare_attack"] and (last_ok or not real):
            mode = "ATTACK"
        else:
            mode = "NORMAL"
    elif drought >= 2 or (cover > 1.0 and rounds_left > 1):         # TAC_RISK=0: first version (cover only)
        mode = "RESCUE"
    elif drought == 1 or cover > p["cover_safe"] or rounds_left <= 1:
        mode = "SAFE"
    elif cover < p["cover_attack"] and (last_ok or not real):
        mode = "ATTACK"
    else:
        mode = "NORMAL"
    if rounds_left <= 1 and stock > cap and os.environ.get("TAC_LAST_RESCUE", "1") == "1":
        mode = "RESCUE"                        # stage 2 R8: 11 bags left for 5 slots, unsold = 0.80 anyway
    if risk_mode == "2":                       # cover modes + sell-limit override when the danger is large
        if r["unsold"] > p["unsold_override"] and rounds_left > 1:
            mode = "RESCUE"
        elif r["if_miss"] > p["miss_override"] and mode in ("NORMAL", "ATTACK"):
            mode = "SAFE"
    order = ["RESCUE", "SAFE", "NORMAL", "ATTACK"]
    if prev in order and order.index(mode) > order.index(prev) + 1:      # climb one step per round
        mode = order[order.index(prev) + 1]
    return mode, {"cover": round(cover, 2), "drought": drought, "stock": stock, "cap": cap, **r}


def apply(state: dict, ctx, rows: list[dict], planner_rows: list[dict], rounds_left: int, practice: bool) -> list[dict]:
    """rows = the plan about to be guarded (LLM or planner); planner_rows = the planner's own plan (for the LLM clamp
    and for filling capacity). Returns new rows; sets ctx._k_floor for the guard."""
    p = params()
    nxt, nxt_type, cap_b, hist, extra = feedback(ctx.memory, ctx.tv, p)
    try:
        sellable = sellable_lot_ids(state, ctx.memory)
    except Exception:                          # noqa: BLE001 - fall back to all lots
        sellable = None
    mode, info = choose_mode(state, hist, rounds_left, p, getattr(ctx.memory, "tactic_mode", None), sellable)
    if practice:
        mode = "NORMAL"                         # practice rounds: learn ceilings at the planner's prices
    if ctx.memory is not None and not practice:
        ctx.memory.tactic_mode = mode
    rfqs = {r["rfq_id"]: r for r in state["rfqs"]}
    plan_k = {(r["lot_id"], r["rfq_id"]): float(r["markup"]) for r in planner_rows}
    floor = SALVAGE + p["floor_gap"]
    ctx._k_floor = floor if (mode == "RESCUE" or rounds_left <= 1) and not practice else None
    shift = p["shift"][mode]
    # TAC_ANCHOR: 0 off · 1 always · 2 only after it stopped working (SAFE / RESCUE: a dry round or behind pace)
    anchor_mode = os.environ.get("TAC_ANCHOR", "0")      # suite + stage-2 replay 19 Nov: anchor biased, off
    use_anchor = anchor_mode == "1" or (anchor_mode == "2" and mode in ("SAFE", "RESCUE"))
    anc = anchors(ctx.memory, state["lots"], ctx.tv, p) if use_anchor else {}
    out, changed = [], []
    lot_floor = {}
    use_wr = os.environ.get("TAC_WINRATE", "0") == "1"
    wr_ev = win_evidence(ctx.memory, ctx.tv) if use_wr else {}
    rows = list(rows)
    reserve = None
    if os.environ.get("TAC_RESERVE", "0") == "1" and not practice and mode in ("NORMAL", "ATTACK") and rounds_left >= 3:
        total = sum(l.get("kg_total", l["kg_available"]) for l in state["lots"]) // BAG
        pr = rival_pressure(ctx.memory, getattr(ctx, "n_teams", None), total, rounds_left)
        if pr is not None and pr < p["res_pressure"] and info["cover"] < p["res_cover"]:
            # rivals will run short: sell an even share now (not the whole capacity) and ask a bit more
            reserve = {"pressure": pr, "bags": max(1, -(-info["stock"] // rounds_left))}
            shift += p["res_premium"]
    if mode in ("SAFE", "RESCUE"):              # fill the round: planner rows the plan left out go after it
        have = {(r["lot_id"], r["rfq_id"]) for r in rows}
        rows += [r for r in planner_rows if (r["lot_id"], r["rfq_id"]) not in have]
    for r in rows:
        q = rfqs.get(r["rfq_id"])
        if q is None:
            continue
        buyer, btype = q.get("buyer", "?"), _type(q.get("buyer_type"))
        k0 = float(r["markup"])
        k = k0
        pk = plan_k.get((r["lot_id"], r["rfq_id"]))
        if pk is not None and abs(k - pk) > p["llm_delta"]:              # 3. LLM clamp
            k = pk + (p["llm_delta"] if k > pk else -p["llm_delta"])
        fb = nxt.get(buyer, nxt_type.get(btype))
        if fb is not None:                                               # 1. feedback replaces the guess
            k = fb
        k += shift                                                       # 2. mode
        a = anc.get(buyer)
        if a is not None and use_anchor:                                 # 6. never far above what the buyer paid
            k = min(k, a["k"] + p["premium"][mode])
        if buyer in cap_b:
            k = min(k, cap_b[buyer])
        elif btype in extra["type_cap"] and buyer not in nxt:     # new buyer: its type was refused at this k
            k = min(k, extra["type_cap"][btype])
        if use_wr and not practice and wr_ev.get(btype):                # 8. win-rate pricing (TAC_WINRATE=1)
            acc = [kk for kk, w in wr_ev[btype] if w]
            keep = SALVAGE if (info["cover"] > 0.85 or rounds_left <= 1) else max(SALVAGE, (sum(acc) / len(acc) - p["wr_keep_gap"]) if acc else SALVAGE)
            grid = [round(k - 0.06 + 0.005 * i, 4) for i in range(21)]
            k = max(grid, key=lambda x: (x - keep) * win_prob(x, wr_ev[btype], k))
            if buyer in cap_b:
                k = min(k, cap_b[buyer])
        if rounds_left <= 1 and not practice:   # last round: unsold = 0.80, so price under what this type took lately
            recent = [kk for i, kk in extra["type_acc"].get(btype, []) if i >= extra["rounds"] - 3]
            if recent:
                k = min(k, max(recent) - p["last_gap"])
        f = extra["lot_factor"].get(r["lot_id"], 1.0) if os.environ.get("TAC_LOT_TV", "0") == "1" else 1.0
        if f < 1.0:                                                      # 7. lot TV correction
            k *= f
            lot_floor[r["lot_id"]] = round(floor * f, 4)
        k = max(k, floor * f)                                            # 5. floor (in the lot's TRUE terms)
        k = round(k, 4)
        if abs(k - k0) > 1e-4:
            changed.append(f"{r['rfq_id']}:{k0:.3f}->{k:.3f}")
        out.append({**r, "markup": k})
    if reserve:                                                          # 9. keep stock for the late rounds
        room, kept = reserve["bags"], []
        for r in sorted(out, key=lambda r: -r["markup"]):
            b = min(int(r["bags"]), room)
            if b > 0:
                kept.append({**r, "bags": b})
                room -= b
        out = kept
        ctx.notes.append(f"reserve: rivals' stock / demand left {reserve['pressure']} -> offer {reserve['bags']} bags now")
    ctx._lot_floor = lot_floor                                           # guard: these lots may go under 0.85
    out = spread(out, state, p)                                          # 4. spread
    ctx.tactic = {"mode": mode, **info, "caps": {b: round(v, 3) for b, v in cap_b.items()},
                  "anchors": {b: v["k"] for b, v in anc.items()},
                  "lot_factor": {l: round(v, 3) for l, v in extra["lot_factor"].items()},
                  "next_k": {b: round(v, 3) for b, v in nxt.items()}, "changes": changed[:12]}
    ctx.notes.append(f"tactic {mode} (stock {info['stock']} bags, cap {info['cap']}/round, sell-through {info['rate']}, "
                     f"needs {info['need']} of {rounds_left} rounds, expected unsold {info['unsold']:.0%}, "
                     f"if this round misses {info['if_miss']:.0%}, drought {info['drought']}); "
                     f"caps {ctx.tactic['caps']}; k changes {', '.join(changed[:10]) or 'none'}")
    return out


def spread(rows: list[dict], state: dict, p: dict) -> list[dict]:
    """No single offer takes more than SPREAD of the round capacity when other RFQs exist in the plan."""
    cap_kg = state.get("capacity_left_kg")
    if cap_kg is None or len({r["rfq_id"] for r in rows}) < 2:
        return rows
    most = max(1, int(cap_kg // BAG * p["spread"] + 0.999))
    per_rfq = defaultdict(int)
    out = []
    for r in rows:
        bags = min(int(r["bags"]), most - per_rfq[r["rfq_id"]])
        if bags <= 0:
            continue
        per_rfq[r["rfq_id"]] += bags
        out.append({**r, "bags": bags})
    return out
