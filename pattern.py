"""Team pattern (18 Nov): sell into the market's GAPS, each lot to the buyers it FITS, without rushing.

PATTERN=1 in .env turns it on (default off: the planner in planner.py decides). Ideas, from the team and the live
Day-18 practice data (5 games, k on true TV):

  * Buyers and their specs are fixed (B1..B6). Each lot "fits" the buyers whose spec it passes with p >= 0.9; its
    value at a buyer is the k we expect to get there. A lot is offered only to the buyers within PATTERN_TOL of its
    best fit (focused, not an all-rounder) - e.g. a Specialty-grade lot does not go to Instant at 0.92.
  * Price comes from the market gap, not from k* alone:
      gap buyers (scarce: offers < wanted, or ceiling-bound: wanted left unfilled although offers came) -> just under
      the highest k this buyer accepted, or CLIMB +PATTERN_CLIMB while it never cut us (ceiling still unknown);
      crowded buyers (price war, similar specs) -> k* = argmax (k - V) P(k) from the learnt ceiling + competition.
  * No rush: the round capacity already caps how much we can sell, and unsold stock is rare (organisers), so lots wait
    for their best buyer. Safety: when the stock left exceeds what the remaining rounds can place, and in the last
    PATTERN_LATE rounds, every fitting buyer opens and crowded-buyer prices are used.
  * Floor: never below K_FLOOR (and never at/below the 0.80 salvage); a small random undercut avoids exact ties.

Returns the same plan rows as strategy.rule_plan: [{"lot_id", "rfq_id", "bags", "markup", "p"}]. guard() still
clamps every number afterwards, and the LLM editor (if on) edits this plan.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict

import lpalloc
import market_core as mc
from pricing import SALVAGE, best_k, norm_type

BAG = 60
# type defaults before any round has been seen (live practice: SPE 0.4-1.0 offers/RFQ, COM ~7, INS ~20)
TYPE_KIND = {"SPECIALTY": "scarce", "COMMERCIAL": "balanced", "INSTANT": "crowded"}


def env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def buyer_kinds(rounds: list[dict], last: int = 3) -> dict[str, dict]:
    """Per buyer NAME, from the last cleared rounds: scarce / ceiling / crowded / balanced (position.buyer_gaps rule).
    RIVAL offers only: our own offers on a scarce buyer must not make it look crowded (review 13:20)."""
    stats = defaultdict(lambda: [0, 0, 0])
    for rnd in rounds[-last:]:
        own = defaultdict(int)
        for o in rnd.get("my_offers", []):
            own[o["rfq_id"]] += 1
        for row in rnd.get("market_summary", []):
            s = stats[row["buyer"]]
            s[0] += row["wanted_kg"] // BAG
            s[1] += row["bought_kg"] // BAG
            s[2] += max(0, (row.get("offers") or 0) - own[row["rfq_id"]])
    out = {}
    for name, (w, b, n) in stats.items():
        ratio = n / max(w, 1)
        kind = "scarce" if ratio < 1 else "ceiling" if b < w else "crowded" if ratio >= 1.3 else "balanced"
        out[name] = {"kind": kind, "offers_per_wanted": round(ratio, 2), "fill": round(b / max(w, 1), 2)}
    return out


TYPE_OFFERS = {"SPECIALTY": 1.0, "COMMERCIAL": 7.0, "INSTANT": 20.0}   # live offers per RFQ before any history


def horizon_demand(rounds: list[dict], rfqs: list[dict], buyers: dict, left_rounds: int, share_k: float,
                   last: int = 3) -> dict[str, int]:
    """Bags each buyer can still take FROM US over the remaining rounds: bags it wants per round x our share, where
    share = min(0.6, share_k / offers per RFQ) (more rival offers on its RFQs -> smaller slice for us)."""
    want, offers, n_rfq, seen = defaultdict(float), defaultdict(float), defaultdict(int), set()
    for rnd in rounds[-last:]:
        seen.add(rnd.get("round_id"))
        for row in rnd.get("market_summary", []):
            want[row["buyer"]] += row["wanted_kg"] / BAG
            offers[row["buyer"]] += row.get("offers") or 0
            n_rfq[row["buyer"]] += 1
    n_rounds = max(1, len(seen))
    now = defaultdict(float)
    for r in rfqs:
        now[r["buyer"]] += r["bags"]
    out = {}
    for name, (btype, _spec) in buyers.items():
        if n_rfq[name]:
            per_round, per_rfq_offers = want[name] / n_rounds, offers[name] / n_rfq[name]
        else:
            per_round, per_rfq_offers = now.get(name, 0.0), TYPE_OFFERS.get(btype, 5.0)
        share = min(0.6, share_k / max(1.0, per_rfq_offers))
        out[name] = int(round(per_round * share * left_rounds))
    return out


def buyer_k(name: str, btype: str, kind: str, learner, floor: float, climb: float, edge: float,
            min_p: float = 0.0) -> tuple[float, float]:
    """(k to ask, rough P) at this buyer. Gap buyers: at the edge of what it accepted; else k* with competition."""
    b = learner.belief(name, btype)
    ev = learner.by_name.get(name)
    if kind in ("scarce", "ceiling"):
        passed = max(ev.passed) if ev and ev.passed else None
        above = min(ev.above) if ev and ev.above else None
        if passed is None:
            k = b.lo                                     # nothing accepted yet: start at the low end of the band
        elif above is None:
            k = passed + climb                           # never cut us: climb to find the ceiling
        else:
            k = min(passed, above) - edge                # ceiling found: sit just under it
        k = min(max(k, floor), b.hi)
        return k, 0.8 if passed is not None else 0.6
    r = best_k(b, reserve=SALVAGE, floor=floor, min_p=min_p)
    return r["k"], r["p"]


def plan(state: dict, tv: dict, learner, rounds: list[dict], known_buyers: dict | None, rules: dict, noise,
         rng, notes: list, bans: set | None = None) -> list[dict]:
    rnd, total = state["round"], rules["timeline"]["rounds"]
    number = 1 if rnd["kind"] == "practice" else rnd["number"]
    left_rounds = total - number + 1
    floor = max(env_f("K_FLOOR", 0.85), SALVAGE + 0.01)
    tol, late = env_f("PATTERN_TOL", 0.03), int(env_f("PATTERN_LATE", 2))
    climb, edge, undercut = env_f("PATTERN_CLIMB", 0.01), env_f("PATTERN_EDGE", 0.005), env_f("UNDERCUT", 0.01)
    min_p = 0.9

    kinds = buyer_kinds(rounds)
    buyers: dict[str, tuple[str, dict]] = {n: (norm_type(b["type"]), b["spec"]) for n, b in (known_buyers or {}).items()}
    for r in state["rfqs"]:
        buyers[r["buyer"]] = (norm_type(r["buyer_type"]), r["spec"])

    def kind_of(name, btype):
        return kinds.get(name, {}).get("kind") or TYPE_KIND.get(btype, "balanced")

    ask = {n: buyer_k(n, t, kind_of(n, t), learner, floor, climb, edge) for n, (t, _s) in buyers.items()}
    # live 18 Nov 12:36: an Instant-only lot offered at k 0.85 in R7/R8 was OUTBID both times and salvaged at 0.80.
    # At the end anything above 0.80 beats salvage: the last round (and the one before, for lots that only fit
    # crowded buyers) may go down to PATTERN_LASTFLOOR.
    last_floor = max(SALVAGE + 0.005, env_f("PATTERN_LASTFLOOR", 0.81))
    # ... and at the end the price must actually SELL (live 12:39: k* 0.877 in R8 was OUTBID -> salvage): crowded
    # buyers get the highest k with P(sell) >= PATTERN_LATE_P
    late_p = env_f("PATTERN_LATE_P", 0.9)
    ask_late = {n: buyer_k(n, t, kind_of(n, t), learner, last_floor, climb, edge, min_p=late_p)
                for n, (t, _s) in buyers.items()}

    lots = [l for l in state["lots"] if l["kg_available"] >= BAG]
    stock = sum(l["kg_available"] for l in lots) // BAG
    cap = state.get("capacity_left_kg")
    cap_bags = cap // BAG if cap is not None else stock
    use_lp = os.environ.get("PATTERN_ALLOC", "fit") == "lp"
    # safety valve: more stock than the remaining rounds can place at ~60 % fill -> open every fitting buyer
    # (the LP variant handles this itself through its capacity constraint)
    behind = stock > cap_bags * left_rounds * 0.6
    open_all = left_rounds <= late or (behind and not use_lp)
    if behind and not use_lp:
        notes.append(f"pattern: stock {stock} bags > what {left_rounds} rounds x {cap_bags} can place -> open all")

    fit: dict[str, dict[str, float]] = {}
    for l in lots:
        vals = {n: ask[n][0] for n, (t, spec) in buyers.items()
                if mc.pass_probability(l["sample"], spec, noise) >= min_p and (l["lot_id"], n) not in (bans or set())}
        fit[l["lot_id"]] = vals
    best = {lid: max(v.values()) if v else None for lid, v in fit.items()}

    assigned: dict[str, dict[str, int]] = {}
    if use_lp:
        demand = horizon_demand(rounds, state["rfqs"], buyers, left_rounds, env_f("PATTERN_SHARE", 2.0))
        value = {(l["lot_id"], n): (ask[n][0] - SALVAGE) * ask[n][1] * tv[l["lot_id"]] * BAG
                 for l in lots for n in fit[l["lot_id"]]}
        assigned = lpalloc.assign({l["lot_id"]: l["kg_available"] // BAG for l in lots}, demand, value,
                                  cap_bags * left_rounds)
        notes.append("pattern lp: demand " + ", ".join(f"{n.split()[0]}={d}" for n, d in sorted(demand.items()))
                     + " | plan " + "; ".join(f"{lid[-1]}->" + ",".join(f"{b.split()[0]}:{n}" for b, n in a.items())
                                              for lid, a in sorted(assigned.items())))

    # Sell risk (live practice 18 Nov, 78 revealed lots): lots that only fit crowded Instant buyers sold nothing at all
    # in 27 % of games (Commercial-grade 6 %, Specialty-grade 0 %) and fetched k 0.88. PATTERN_RISK=1: such lots are
    # priced PATTERN_RISK_CUT lower, get PATTERN_RISK_PACE slots per round and are served first in the quota pass.
    use_risk = os.environ.get("PATTERN_RISK", "0") == "1"
    risk_cut, risk_pace = env_f("PATTERN_RISK_CUT", 0.02), env_f("PATTERN_RISK_PACE", 2.0)
    risky = {lid for lid, v in fit.items() if v and all(kind_of(n, buyers[n][0]) == "crowded" for n in v)}
    if use_risk and risky:
        notes.append(f"pattern risk: {sorted(risky)} only fit crowded buyers -> k -{risk_cut}, pace {risk_pace}")

    cands = []
    for r in state["rfqs"]:
        name, btype = r["buyer"], norm_type(r["buyer_type"])
        k, p = ask[name]
        for l in lots:
            lid = l["lot_id"]
            if name not in fit[lid] or mc.pass_probability(l["sample"], r["spec"], noise) < min_p:
                continue
            if not open_all and use_lp and assigned.get(lid, {}).get(name, 0) <= 0:
                continue                                  # the horizon LP sends this lot to other buyers
            if not open_all and not use_lp and fit[lid][name] < best[lid] - tol:
                continue                                  # a better-fitting buyer exists: keep the lot for it
            kk = k
            only_crowded = all(kind_of(n, buyers[n][0]) == "crowded" for n in fit[lid])
            if left_rounds <= 1 or (left_rounds <= 2 and only_crowded):
                kk = ask_late[name][0]
            if use_risk and lid in risky:                 # hard-to-sell lot: trade a little k for a sale
                kk = max(min(floor, kk), kk - risk_cut)
            if open_all and kind_of(name, btype) in ("scarce", "ceiling") and left_rounds <= 1:
                kk = max(last_floor, min(kk, learner.belief(name, btype).lo))   # last round: sure sale at a gap buyer
            cands.append({"lot": l, "rfq": r, "k": kk, "p": p, "value": (kk - SALVAGE) * p * tv[lid]})

    lot_left = {l["lot_id"]: l["kg_available"] // BAG for l in lots}
    rfq_left = {r["rfq_id"]: r["bags"] for r in state["rfqs"]}
    room = cap_bags
    out = []
    # live game 18 Nov 11:41: one 7-bag offer on one RFQ per round (all-or-nothing OUTBID) and the best lot taking
    # every slot left 3 lots never offered (62 % salvaged). So: (1) each lot gets its share of the round's slots
    # first (~ PATTERN_PACE x its stock / rounds left), the rest of the room goes by value; (2) one offer takes at
    # most PATTERN_SPREAD of the round's capacity.
    pace, spread = env_f("PATTERN_PACE", 1.5), env_f("PATTERN_SPREAD", 0.5)
    quota = {lid: max(1, -(-int(n * (risk_pace if use_risk and lid in risky else pace)) // max(1, left_rounds)))
             for lid, n in lot_left.items()}
    # same rule as guard(): ceil(capacity x spread); the two passes together never exceed it on one (lot, RFQ)
    per_offer = max(1, math.ceil(cap_bags * spread)) if left_rounds > 1 else cap_bags
    placed: dict[tuple[str, str], int] = defaultdict(int)
    order = sorted(cands, key=lambda c: (kind_of(c["rfq"]["buyer"], norm_type(c["rfq"]["buyer_type"])) == "crowded",
                                         -c["value"], -tv[c["lot"]["lot_id"]]))
    # PATTERN_ORDER=options: quota pass serves lots with the fewest fitting buyers first. Engine 12:45 (10 games x 5,
    # paired): -0.0100 PR vs by value (thin 0.876 vs 0.915) -> default stays "value".
    by_options = order
    if os.environ.get("PATTERN_ORDER", "value") == "options":
        by_options = sorted(order, key=lambda c: len(fit[c["lot"]["lot_id"]]))
    elif use_risk:
        by_options = sorted(order, key=lambda c: c["lot"]["lot_id"] not in risky)   # stable: value order kept
    for use_quota in (True, False):
        for c in (by_options if use_quota else order):
            lid, rid = c["lot"]["lot_id"], c["rfq"]["rfq_id"]
            bags = min(lot_left[lid], rfq_left[rid], room, per_offer - placed[(lid, rid)])
            if use_quota:
                bags = min(bags, quota[lid])
            if bags <= 0:
                continue
            k = max(min(floor, c["k"]), c["k"] - rng.uniform(0, undercut))
            out.append({"lot_id": lid, "rfq_id": rid, "bags": bags, "markup": round(k, 4), "p": round(c["p"], 2)})
            lot_left[lid] -= bags
            rfq_left[rid] -= bags
            quota[lid] -= bags
            placed[(lid, rid)] += bags
            room -= bags
            if room <= 0:
                break
        if room <= 0:
            break
    merged = {}
    for o in out:                                       # same lot + RFQ from both passes -> one row
        key = (o["lot_id"], o["rfq_id"])
        if key in merged:
            merged[key]["bags"] += o["bags"]
        else:
            merged[key] = o
    out = list(merged.values())
    kinds_now = {n: kind_of(n, t) for n, (t, _s) in buyers.items()}
    notes.append("pattern: " + ", ".join(f"{n.split()[0]}={kinds_now[n]}@{ask[n][0]:.3f}" for n in sorted(buyers))
                 + f" | open_all={open_all} offers={len(out)} bags={sum(o['bags'] for o in out)}/{cap_bags}")
    return out
