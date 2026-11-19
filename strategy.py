"""Decide this round's offers - Track 3-2: the LLM is the seller.

Every round:
  1. facts()       Python computes what a small model cannot: spec-checked (lot x RFQ) candidates, TV, the learnt
                   ceiling band per buyer, a suggested k, and per-buyer market stats from earlier rounds.
  2. the event LLM decides the offers, LLM_MODE picks how (.env):
       editor (default)  llm_edit(): starts from the planner's plan; keeps / drops / resizes rows, moves k, or holds
       seller            llm_plan(): picks every offer itself from the candidate list
     (arena, Day 18, seed 0, qwen3.5:9b: seller PR 0.9356 vs rules-only 0.9758 -> editor is the default)
  3. guard()       Python clamps every number: k inside [K_FLOOR, learnt ceiling top], bags <= stock / RFQ /
                   round capacity, spec with noise margin. The LLM can choose badly but never send an invalid
                   or below-salvage offer.
  4. fallback      LLM off / too slow / bad JSON -> the planner's offers (slide 29: must sell with USE_LLM=0).
                   Last round: unsold stock the LLM left is topped up by the planner (else it salvages at 0.80).

price_per_kg = k x our TV estimate. Public names match the starter, so main.py is unchanged.
"""
import json
import math
import os
import random
import re
import zlib
from dataclasses import dataclass, field

import market_core as mc
import pattern
import position as pos
from planner import PlanConfig, plan_round, round_labels, is_festival
from pricing import CeilingLearner, best_k, k_grid, norm_type

BAG = 60
MIN_MARKUP, MAX_MARKUP = 0.85, 1.30
# The event gateway allows ~4,000 tokens per call (prompt + answer). Keep the prompt inside what is left.
LLM_CONTEXT = int(os.environ.get("LLM_CONTEXT", "4000"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "700"))
PROMPT_BUDGET = LLM_CONTEXT - LLM_MAX_TOKENS - 300          # 300 = tokenizer-estimate safety margin
MAX_CANDS = 24
P_MIN = float(os.environ.get("LLM_P_MIN", "0.5"))          # the LLM may not bid where P(sell) < this


def _noise(rules: dict) -> dict | None:
    """Sample noise for spec margins and appraisal. SAMPLE_NOISE="defect,moisture[,screen]" overrides (screen = chance
    the true screen is one size off); on by default when capacity is on (Day 18+)."""
    env = os.environ.get("SAMPLE_NOISE")
    if env:
        parts = [float(x) for x in env.split(",")]
        d, m, s = (parts + [0.0, 0.0, 0.0])[:3]
        return {"defect": d, "moisture": m, "screen": s} if d or m or s else None
    return {"defect": 0.35, "moisture": 0.30} if rules.get("round_capacity") else None


def plan_config(rules: dict) -> PlanConfig:
    return PlanConfig(noise=_noise(rules), floor=float(os.environ.get("K_FLOOR", "0.85")),
                      # lab.py suite (17 Nov): patience 0 + random undercut ("fastcut") keeps rank when many teams
                      # share the market; the old 0.6 / no undercut fell to mid-room. Env vars restore it.
                      patience=float(os.environ.get("PATIENCE", "0.0")), undercut=float(os.environ.get("UNDERCUT", "0.01")),
                      probes=os.environ.get("PROBES", "1") == "1",
                      ladder=os.environ.get("LADDER", "0") == "1")     # 1 bag near the ceiling, rest at k*


@dataclass
class Context:
    rules: dict
    calendar: list
    tv: dict                          # lot_id -> our TV estimate (baht/kg)
    rng: random.Random
    memory: object = None
    llm: object = None
    seconds_left: float = 999
    notes: list = field(default_factory=list)
    n_teams: int | None = None        # sellers in the room (leaderboard size), for market supply


def spec_ok(sample: dict, spec: dict, rules: dict | None = None) -> bool:
    # SPEC_P: how sure the sample must make us that the TRUE lot passes (SPEC_MISMATCH only costs that slot).
    # 0.5 since 20 Nov 00:40: blind 700 markets PR +0.0084, top 5 64 -> 78 %; live practice 9 games PR 0.956,
    # rank 1 in 7, only 3 SPEC_MISMATCH (0.9 was too strict: self-play rooms all crowded Instant).
    return mc.pass_probability(sample, spec, _noise(rules or {})) >= float(os.environ.get("SPEC_P", "0.5"))


def rounds_left(state: dict, ctx: Context) -> int:
    rnd, total = state["round"], ctx.rules["timeline"]["rounds"]
    return total if rnd["kind"] == "practice" else total - rnd["number"] + 1


def _round_numbers(state: dict, ctx: Context) -> tuple[int, int]:
    rnd, total = state["round"], ctx.rules["timeline"]["rounds"]
    return (1, total) if rnd["kind"] == "practice" else (rnd["number"], total)


def _stricter(a: dict, b: dict) -> bool:
    """Spec a is at least as strict as spec b on every limit."""
    def lim(s, k, default):
        v = s.get(k)
        return default if v is None else v
    return (lim(a, "max_defect_pct", 100) <= lim(b, "max_defect_pct", 100)
            and lim(a, "max_moisture_pct", 100) <= lim(b, "max_moisture_pct", 100)
            and lim(a, "min_screen", 0) >= lim(b, "min_screen", 0))


def spec_bans(ctx: Context) -> set[tuple[str, str]]:
    """(lot_id, buyer) pairs to skip for the rest of the game: a SPEC_MISMATCH means the lot's TRUE quality fails
    that buyer's spec, so it fails every buyer at least as strict too. Live 12:51: a lot (sample defect 2.6, true
    3.16) was offered to the same Specialty buyer 7 rounds in a row and never sold anywhere."""
    memory = getattr(ctx, "memory", None)
    if memory is None:
        return set()
    specs = {n: b["spec"] for n, b in (getattr(memory, "buyers", None) or {}).items()}
    bans = set()
    for rnd in memory.rounds:
        buyer_of = {r["rfq_id"]: r.get("buyer") for r in rnd.get("market_summary", [])}
        for o in rnd.get("my_offers", []):
            if o.get("status") != "SPEC_MISMATCH":
                continue
            failed = buyer_of.get(o["rfq_id"])
            if not failed:
                continue
            bans.add((o["lot_id"], failed))
            for name, spec in specs.items():
                if failed in specs and _stricter(spec, specs[failed]):
                    bans.add((o["lot_id"], name))
    return bans


def learner_from_memory(state: dict, ctx: Context) -> CeilingLearner:
    # JITTER = how far one buyer's ceiling moves between its RFQs (lo = highest passed k - jitter).
    # 0.025 = organiser engine; 17 Nov practice data showed +-1-1.5 % per buyer.
    learner = CeilingLearner(jitter=float(os.environ.get("JITTER", "0.025")))
    samples = {l["lot_id"]: l["sample"] for l in state["lots"]}
    for rnd in (ctx.memory.rounds if ctx.memory else []):
        learner.observe(rnd, ctx.tv, samples)
    return learner


def dynamic_patience(base: float, pressure: float | None) -> float:
    """Dynamic PR aim, expressed as the planner's patience (how much of the hold value V counts).

    k* = argmax (k - V) x P(k): a higher V pushes k* up toward the learnt ceiling, a lower V sells now.
    pressure = market supply left / demand left (position.market_balance):
      >= 1.1  sellers outnumber the bags wanted -> prices fall, sell now (base patience)
      <= 0.6  buyers will be short -> wait for the price, patience 0.8 (near the ceiling)
    in between: linear. No team count -> pressure unknown -> base."""
    if pressure is None:
        return base
    return round(base + (0.8 - base) * min(1.0, max(0.0, (1.1 - pressure) / 0.5)), 3)


def dynamic_labels(ctx: Context, total: int) -> dict[int, str]:
    """Calendar labels, plus 'festival' for any round whose demand_weight is >= 1.15 (the live calendar gives a
    number even when the label is just a name), so the planner keeps stock for high-demand rounds."""
    labels = round_labels(ctx.calendar, total)
    for n, (w, _share) in pos.calendar_weights(ctx.calendar, total).items():
        if w >= 1.15:
            labels[n] = "festival"
    return labels


def _gap_push(state: dict, ctx: Context, plan, learner) -> None:
    """Buyers whose last rounds show a gap (ceiling-bound: bought < wanted although enough offers; or scarce:
    offers < wanted) are not limited by competition, so k* = argmax (k - V) P undersells them. Price those offers
    just under the highest k that buyer has ever accepted from us (- GAP_EDGE), never above the learnt top."""
    try:
        gaps = position_facts(state, ctx)["gaps"]
    except Exception as e:                           # noqa: BLE001
        ctx.notes.append(f"gap push skipped: {e!r}")
        return
    edge = float(os.environ.get("GAP_EDGE", "0.005"))
    rfqs = {r["rfq_id"]: r for r in state["rfqs"]}
    for o in plan.offers:
        r = rfqs.get(o.rfq_id)
        if r is None or gaps.get(buyer_code(r), {}).get("kind") not in ("ceiling", "scarce"):
            continue
        ev = learner.by_name.get(r["buyer"])
        if not ev or not ev.passed:
            continue
        top = learner.belief(r["buyer"], r["buyer_type"]).hi
        k_new = min(max(ev.passed) - edge, top)
        if k_new > o.k:
            ctx.notes.append(f"gap push {o.rfq_id} {buyer_code(r)}: k {o.k:.3f} -> {k_new:.3f}")
            o.k = round(k_new, 4)


def _hold_back(state: dict, ctx: Context, plan, number: int, total: int) -> None:
    """Two hold rules, both off by default, never in the last round (unsold stock then salvages at 0.80).
    SPEC_HOLD=1: a lot whose best reachable buyer type is Specialty (position.allocation_pull) is offered only to
                 Specialty RFQs until the last round.
    LATE_KEEP=f: before the last two rounds, keep at least f x our starting bags unoffered (drop the lowest-k
                 offers first) so we still have stock when rivals have run out late in the game."""
    if number >= total:
        return
    if os.environ.get("SPEC_HOLD", "0") == "1":
        try:
            pull = position_facts(state, ctx)["pull"]
            types = {r["rfq_id"]: norm_type(r["buyer_type"]) for r in state["rfqs"]}
            keep = [o for o in plan.offers
                    if not (pull.get(o.lot_id, {}).get("hold_for_specialty") and types.get(o.rfq_id) != "SPECIALTY")]
            if len(keep) < len(plan.offers):
                ctx.notes.append(f"spec hold: dropped {len(plan.offers) - len(keep)} non-Specialty offers")
            plan.offers = keep
        except Exception as e:                       # noqa: BLE001
            ctx.notes.append(f"spec hold skipped: {e!r}")
    keep_share = float(os.environ.get("LATE_KEEP", "0") or 0)
    if keep_share > 0 and number < total - 1:
        start = sum(l.get("kg_total", l["kg_available"]) for l in state["lots"]) // BAG
        stock = sum(l["kg_available"] for l in state["lots"]) // BAG
        room = max(0, stock - int(keep_share * start + 0.5))
        out = []
        for o in sorted(plan.offers, key=lambda o: -o.k):
            o.bags = min(o.bags, room)
            room -= o.bags
            if o.bags > 0:
                out.append(o)
        if len(out) < len(plan.offers) or room == 0:
            ctx.notes.append(f"late keep {keep_share}: offering {sum(o.bags for o in out)} of {stock} bags")
        plan.offers = out


def rule_plan(state: dict, ctx: Context, cands: list[dict] | None = None) -> list[dict]:
    """The planner (fallback and last-round top-up): learnt ceilings + hold value + greedy allocation."""
    learner = getattr(ctx, "_learner", None) or learner_from_memory(state, ctx)
    number, total = _round_numbers(state, ctx)
    cfg = plan_config(ctx.rules)
    labels = round_labels(ctx.calendar, total)
    # off by default: lab suite 18 Nov (6 games x 4 markets) - static agent avg rank 23.7 %, dynamic 24.6 %
    # (crowd d18 34.8 % -> 37.1 %); solo arena 12 games PR 0.9611 vs 0.9610. Retest on live data before enabling.
    if os.environ.get("DYN_PR", "0") == "1":
        try:
            pressure = position_facts(state, ctx)["balance"].get("pressure")
            cfg.patience = dynamic_patience(cfg.patience, pressure)
            labels = dynamic_labels(ctx, total)
            ctx.notes.append(f"dynamic PR: pressure {pressure} -> patience {cfg.patience}")
        except Exception as e:                       # noqa: BLE001 - fall back to the static planner
            ctx.notes.append(f"dynamic PR failed: {e!r}")
    known_buyers = None
    memory = getattr(ctx, "memory", None)
    if memory is not None and hasattr(memory, "remember_buyers"):   # every buyer's spec seen this game
        try:
            if memory.remember_buyers(state["rfqs"]):
                memory.save()
            known_buyers = memory.buyers
        except Exception as e:                       # noqa: BLE001 - planning works with this round's RFQs alone
            ctx.notes.append(f"buyer specs not remembered: {e!r}")
    # PLAN_BANS=1: leave SPEC_MISMATCH pairs out before capacity is shared (replica 60 seeds +0.0063 PR, 11 better /
    # 0 worse, p 0.001; engine compare_methods 5x20 games -0.0006, no significant difference: the engine's small
    # noise hardly makes SPEC_MISMATCH). Enabled by the team on 19 Nov on the replica result. PLAN_BANS=0 = old.
    try:
        bans = spec_bans(ctx) if os.environ.get("PLAN_BANS", "1") == "1" else set()
    except Exception as e:                           # noqa: BLE001 - the guard still drops banned pairs
        bans = set()
        ctx.notes.append(f"spec bans not read: {e!r}")
    plan = plan_round(state["lots"], state["rfqs"], ctx.tv, learner, round_number=number, rounds_total=total,
                      capacity_left_kg=state.get("capacity_left_kg"), labels=labels, cfg=cfg,
                      known_buyers=known_buyers, bans=bans)
    ctx._learner, ctx._reserve = learner, plan.reserve
    if os.environ.get("PATTERN", "0") == "1":        # team pattern: gaps + lot/buyer fit (pattern.py)
        try:
            rows = pattern.plan(state, ctx.tv, learner, ctx.memory.rounds if ctx.memory else [], known_buyers,
                                ctx.rules, cfg.noise, ctx.rng, ctx.notes, bans=spec_bans(ctx))
            if plan.unsellable:
                ctx.notes.append(f"unsellable (salvage 0.80): {plan.unsellable}")
            return rows
        except Exception as e:                       # noqa: BLE001 - the planner below still sells
            ctx.notes.append(f"pattern failed, planner used: {e!r}")
    if os.environ.get("GAP_PUSH", "0") == "1":
        _gap_push(state, ctx, plan, learner)
    _hold_back(state, ctx, plan, number, total)
    if plan.unsellable:
        ctx.notes.append(f"unsellable (salvage 0.80): {plan.unsellable}")
    if plan.skipped:                                 # why pairs were not offered (empty rounds are otherwise silent)
        ctx.notes.append("skipped: " + "; ".join(f"{l[-5:]}->{r}: {why}" for l, r, why in plan.skipped[:12]))
    return [{"lot_id": o.lot_id, "rfq_id": o.rfq_id, "bags": o.bags, "markup": o.k, "p": round(o.p, 2)}
            for o in plan.offers]


# ---------------------------------------------------------------------- facts for the LLM

def buyer_code(rfq: dict) -> str:
    """Short ASCII id for a buyer ("3 สยามคอมเมิร์ซ" -> "COM3"): Thai names cost many tokens."""
    m = re.match(r"\s*(\d+)", rfq.get("buyer") or "")
    return norm_type(rfq["buyer_type"])[:3] + (m.group(1) if m else str(zlib.crc32((rfq.get("buyer") or "").encode()) % 100))


def candidates(state: dict, ctx: Context) -> list[dict]:
    """Spec-checked (lot x RFQ) pairs with the numbers the LLM needs. Sorted by expected value, best first."""
    learner = getattr(ctx, "_learner", None) or learner_from_memory(state, ctx)
    ctx._learner = learner
    reserve = getattr(ctx, "_reserve", {})
    number, total = _round_numbers(state, ctx)
    festival = is_festival(round_labels(ctx.calendar, total).get(number))
    floor = float(os.environ.get("K_FLOOR", MIN_MARKUP))
    try:                                         # the LLM must not keep picking pairs guard() will drop
        bans = spec_bans(ctx)
    except Exception:                            # noqa: BLE001
        bans = set()
    out = []
    for rfq in state["rfqs"]:
        b = learner.belief(rfq["buyer"], rfq["buyer_type"], festival)
        for lot in state["lots"]:
            if lot["kg_available"] < BAG or not spec_ok(lot["sample"], rfq["spec"], ctx.rules):
                continue
            if (lot["lot_id"], rfq.get("buyer")) in bans:
                continue
            tv = ctx.tv[lot["lot_id"]]
            V = reserve.get(lot["lot_id"], mc.SALVAGE)
            k = best_k(b, reserve=V, floor=floor)
            # k_hi = highest k that still sells with P >= P_MIN: above it the bid is a near-certain ABOVE_LIMIT/OUTBID
            k_hi = max([x for x in k_grid(floor, MAX_MARKUP) if b.p_accept(x) >= P_MIN] or [floor])
            out.append({"lot_id": lot["lot_id"], "rfq_id": rfq["rfq_id"], "buyer": buyer_code(rfq),
                        "buyer_type": norm_type(rfq["buyer_type"]), "rfq_bags": rfq["bags"],
                        "max_bags": min(lot["kg_available"], rfq["kg"]) // BAG, "tv": tv,
                        # k_lo = floor, not the ceiling's low end: bidding under the ceiling is how you beat the queue
                        "k_lo": round(floor, 3), "k_hi": round(k_hi, 3),
                        "k_sug": round(k["k"], 3), "p_sug": round(k["p"], 2),
                        "ev_bag": (k["k"] - V) * k["p"] * tv * BAG})
    out.sort(key=lambda c: -c["ev_bag"])
    return out


def _real_rounds(ctx: Context, last: int | None = None) -> list[dict]:
    """Cleared scored rounds from memory (the practice round's odd bids would mislead the LLM)."""
    rounds = [r for r in (ctx.memory.rounds if ctx.memory else []) if r.get("kind") != "practice"]
    return rounds[-last:] if last else rounds


def _by_buyer(rnd: dict) -> tuple[dict, dict]:
    """One round's market_summary -> ({rfq_id: buyer code}, {code: [wanted bags, bought bags, offers]})."""
    codes, agg = {}, {}
    for row in rnd.get("market_summary", []):
        code = codes[row["rfq_id"]] = buyer_code(row)
        a = agg.setdefault(code, [0, 0, 0])
        a[0] += row["wanted_kg"] // BAG
        a[1] += row["bought_kg"] // BAG
        a[2] += row.get("offers") or 0
    return codes, agg


def market_stats(ctx: Context, last: int = 2) -> str:
    """Per buyer, the last rounds: bought/wanted and offers per wanted bag (competition pressure)."""
    lines = []
    for rnd in _real_rounds(ctx, last):
        agg = _by_buyer(rnd)[1]
        lines.append(rnd["round_id"] + ": " + "; ".join(
            f"{c} bought {b}/{w} offers/wanted {o / max(w, 1):.1f}" for c, (w, b, o) in sorted(agg.items())))
    return "\n".join(lines) or "none yet"


def my_results(ctx: Context, last: int = 3) -> str:
    """Our own offers in the last rounds as buyer x k -> status (the ceiling / competition evidence)."""
    lines = []
    for rnd in _real_rounds(ctx, last):
        codes = _by_buyer(rnd)[0]
        mine = [f"{codes.get(o['rfq_id'], '?')} k{o['price_per_kg'] / ctx.tv[o['lot_id']]:.3f} {o['status']}"
                for o in rnd.get("my_offers", []) if o["lot_id"] in ctx.tv and o["status"] != "CANCELLED"]
        lines.append(f"{rnd['round_id']}: " + ("; ".join(mine) or "no offers"))
    return "\n".join(lines) or "none yet"


def game_summary(ctx: Context) -> str:
    """Whole game so far, one line per buyer (slide 25 TODO 4): market fill and crowding first -> last round, and
    our evidence about its ceiling (highest k it bought from us, lowest k it rejected as ABOVE_LIMIT)."""
    stats = {}
    for rnd in _real_rounds(ctx):
        codes, agg = _by_buyer(rnd)
        for code, (w, b, n) in agg.items():
            s = stats.setdefault(code, {"w": 0, "b": 0, "first": None, "last": None, "sold": 0, "sent": 0,
                                        "acc": None, "above": None})
            s["w"], s["b"] = s["w"] + w, s["b"] + b
            s["last"] = round(n / max(w, 1), 1)
            if s["first"] is None:
                s["first"] = s["last"]
        for o in rnd.get("my_offers", []):
            s = stats.get(codes.get(o["rfq_id"]))
            if s is None or o["lot_id"] not in ctx.tv or o["status"] == "CANCELLED":
                continue
            k = o["price_per_kg"] / ctx.tv[o["lot_id"]]
            s["sent"] += o["kg"] // BAG
            s["sold"] += (o.get("filled_kg") or 0) // BAG
            if o.get("filled_kg"):
                s["acc"] = k if s["acc"] is None else max(s["acc"], k)
            if o["status"] == "ABOVE_LIMIT":
                s["above"] = k if s["above"] is None else min(s["above"], k)
    lines = []
    for code, s in sorted(stats.items()):
        mine = f"me sold {s['sold']}/{s['sent']}" if s["sent"] else "me none"
        if s["acc"] is not None:
            mine += f" best k{s['acc']:.3f}"
        if s["above"] is not None:
            mine += f" ABOVE at k{s['above']:.3f}"
        lines.append(f"{code}: bought {s['b']}/{s['w']} rivals/bag {s['first']}->{s['last']}; {mine}")
    return "\n".join(lines) or "none yet"


def calendar_ahead(state: dict, ctx: Context, n: int = 3) -> str:
    """This and the next rounds as normal / festival (slide 25 TODO 3), English to save tokens."""
    number, total = _round_numbers(state, ctx)
    labels = round_labels(ctx.calendar, total)
    return ", ".join(f"R{i} {'festival' if is_festival(labels.get(i)) else 'normal'}"
                     for i in range(number, min(total, number + n - 1) + 1))


def position_facts(state: dict, ctx: Context) -> dict:
    """position.py numbers for this round (cached on ctx): lot TV bands, pot, market balance, gaps, lot targets."""
    if getattr(ctx, "_pos", None) is not None:
        return ctx._pos
    noise = _noise(ctx.rules)
    rules = ctx.rules if "defect_steps" in ctx.rules else None      # partial rules -> published defaults
    specs = {buyer_code(r): r["spec"] for r in state["rfqs"]}
    by_type = {}
    for r in state["rfqs"]:
        by_type.setdefault(norm_type(r["buyer_type"]), []).append(buyer_code(r))
    lots = [{**l, "kg_total": l.get("kg_total", l["kg_available"]), "kg_remaining": l.get("kg_remaining", l["kg_available"])}
            for l in state["lots"]]
    bands = {l["lot_id"]: pos.lot_band(l["sample"], rules, noise, specs) for l in lots}
    revenue = (state.get("standings") or {}).get("revenue_thb") or 0
    pot = pos.pot_band(lots, bands, revenue)
    number, total = _round_numbers(state, ctx)
    rounds = _real_rounds(ctx)
    balance = pos.market_balance(rounds, ctx.n_teams, pot["bags_total"], number, total,
                                 pos.calendar_weights(ctx.calendar, total), state["rfqs"])
    type_of = {buyer_code(r): norm_type(r["buyer_type"]) for r in state["rfqs"]}
    own = {c: [b["mid"] for b in bands.values() if b["p_spec"].get(c, 0) >= 0.9] for c in specs}
    implied = pos.implied_k(rounds, specs, buyer_code, lambda c: type_of.get(c, ""), rules, own)
    ctx._pos = {"bands": bands, "pot": pot, "balance": balance, "gaps": pos.buyer_gaps(rounds, buyer_code),
                "pull": pos.allocation_pull(lots, bands, by_type, balance), "implied_k": implied}
    return ctx._pos


def position_text(state: dict, ctx: Context, for_log: bool = False) -> str:
    """The position facts as a few short English lines for the LLM and the decision log ("" if they fail)."""
    try:
        return _position_text(position_facts(state, ctx), for_log)
    except Exception as e:                   # noqa: BLE001 - advice only: a bug here must not cost the round
        ctx.notes.append(f"position failed: {e!r}")
        return ""


def _position_text(p: dict, for_log: bool = False) -> str:
    pot, bal = p["pot"], p["balance"]
    lines = [f"Pot {pot['pot_mid']} THB (TV error {pot['pot_lo']}-{pot['pot_hi']}), bags left {pot['bags_left']}/"
             f"{pot['bags_total']}, PR floor {pot.get('pr_floor')}."]
    if bal.get("pressure") is not None:
        lines.append(f"Market: {bal['n_teams']} sellers, supply left ~{bal['supply_left']} bags vs demand left "
                     f"{sum(bal['demand_left'].values())} (pressure {bal['pressure']}: >1 oversupply, <1 shortage).")
    lines.append("Demand left per type: " + ", ".join(f"{t[:3]} {v}" for t, v in bal["demand_left"].items() if v) + ".")
    if p["gaps"]:
        lines.append("Buyers: " + ", ".join(f"{c} {g['kind']}" for c, g in sorted(p["gaps"].items())) + ".")
    # Not in the LLM prompt by default: arena A/B on qwen3.5:9b, editor, 3 games Day 18 - without 0.9603 (rank 1.3),
    # with 0.9227 (rank 9.3, one game collapsed to rank 20). Always in the decision log (for_log=True).
    if p.get("implied_k") and (for_log or os.environ.get("POS_IMPLIED_K", "0") == "1"):
        lines.append("Avg k others got (last rounds): " + ", ".join(
            f"{c} {'->'.join(f'{k:.2f}' for k in v['k'][-3:])} (+-{v['sd'] * 100:.0f}%)"
            for c, v in sorted(p["implied_k"].items())) + ".")
    lines.append("Lots: " + "; ".join(
        f"{lid[-5:]} -> {v['target'] or 'none'}{' HOLD for SPECIALTY' if v['hold_for_specialty'] else ''}"
        f"{' only outlet' if v['single_outlet'] else ''}, TV err {p['bands'][lid]['err'] * 100:.1f}%"
        for lid, v in p["pull"].items()) + ".")
    return "\n".join(lines)


def ascii_text(text: str) -> str:
    """Prompts are English only: Thai costs several tokens per character on the 4k-context model."""
    return text.encode("ascii", "ignore").decode().strip()


def round_header(state: dict, ctx: Context) -> str:
    rnd, cap = state["round"], state.get("capacity_left_kg")
    return (f"Round {rnd['round_id']} ({rnd['kind']}), rounds left {rounds_left(state, ctx)}. "
            f"Bag cap this round: {cap // BAG if cap is not None else 'none'}.")


def market_notes() -> str:
    notes = ascii_text(os.environ.get("MARKET_NOTES", ""))
    return f"Market notes: {notes}" if notes else ""


# Shared by both system prompts: how to read the whole-game lines.
EVIDENCE = """Whole game per buyer: best k = highest k it bought from us, ABOVE at k = it refused that k (stay below);
rivals/bag = offers per bag wanted (a count ratio, NEVER a k or price); falling (e.g. 1.9->0.9) = supply drying up, ask more. Festival rounds pay more: keep stock for them."""


SYSTEM = """You are the selling agent of a coffee farm in a sealed-offer market with ~56 competing sellers.
k = price / true value. Each buyer drops offers above its hidden ceiling (ABOVE_LIMIT), then buys the lowest k
first until full (OUTBID = others were cheaper). Unsold bags are worth only k=0.80 at the end, so selling at any
k above 0.80 beats keeping a bag after the last round. Earlier rounds: holding is fine if a better k is likely later.
Read the evidence: offers/wanted > 1.3 = crowded, bid near k_lo; < 1.0 or bought < wanted = scarce, bid near k_hi.
ABOVE_LIMIT at k = lower k for that buyer; ACCEPTED every time = you may ask a little more.
Each candidate gives k_lo..k_hi (learnt safe ceiling band), k_sug (model optimum) and ev (expected THB gain per
bag). Candidates are sorted by ev, best first: start from c=0 and fill the bag cap; skip a top candidate only when
your results show that buyer failing. Higher-paying buyer types (SPECIALTY > COMMERCIAL > INSTANT) first.
Stay inside k_lo..k_hi.
""" + EVIDENCE + "\nAnswer only JSON matching the schema."

SCHEMA = {"type": "object", "required": ["offers", "reason"], "properties": {
    "offers": {"type": "array", "items": {"type": "object", "required": ["c", "bags", "k"], "properties": {
        "c": {"type": "integer"}, "bags": {"type": "integer"}, "k": {"type": "number"}}}},
    "reason": {"type": "string", "maxLength": 200}}}


def est_tokens(text: str) -> int:
    """Conservative token estimate. Measured on qwen3.5:9b (18 Nov): number-heavy candidate tables are ~1.75 chars
    per token, prose ~2.7, so divide by 1.5 to stay under the 4,000-token gateway context."""
    return int(len(text) / 1.5) + 1


def fits(system: str, user: str) -> bool:
    return est_tokens(system) + est_tokens(user) <= PROMPT_BUDGET


def llm_prompt(state: dict, ctx: Context, cands: list[dict], plan_bags: int) -> tuple[str | None, list[dict]]:
    """Build the user prompt and shrink it (fewer candidates, less history, then no whole-game lines) until it fits
    PROMPT_BUDGET. None when even the smallest prompt is too big: the caller sells on rules instead of letting the
    gateway reject an over-long request."""
    stock = {l["lot_id"][-5:]: l["kg_available"] // BAG for l in state["lots"]}
    for n_cands, last, whole in ((MAX_CANDS, 3, True), (16, 2, True), (10, 1, True), (6, 1, True), (6, 1, False)):
        shown = cands[:n_cands]
        table = [[i, c["rfq_id"], c["buyer"], c["rfq_bags"], c["lot_id"][-5:], c["max_bags"], round(c["tv"], 1),
                  c["k_lo"], c["k_hi"], c["k_sug"], c["p_sug"], round(c["ev_bag"])] for i, c in enumerate(shown)]
        user = ascii_text("\n".join(filter(None, [
            round_header(state, ctx),
            f"Stock bags per lot: {json.dumps(stock)}. Model suggests offering about {plan_bags} bags now.",
            market_notes(),
            f"Calendar: {calendar_ahead(state, ctx)}.",
            "Position:", position_text(state, ctx),
            *(["Whole game per buyer:", game_summary(ctx)] if whole else []),
            "Market per buyer (last rounds):", market_stats(ctx, min(last, 2)),
            "My offers (last rounds):", my_results(ctx, last),
            "Candidates [c, rfq, buyer, rfq_bags, lot, max_bags, tv, k_lo, k_hi, k_sug, p_sug, ev] "
            "(ev = expected THB gain per bag vs holding; sorted best first):",
            json.dumps(table, separators=(",", ":")),
            "Choose offers: each c at most once, bags <= max_bags, total bags <= bag cap.",
        ])))
        if fits(SYSTEM, user):
            return user, shown
    return None, []


def llm_plan(state: dict, ctx: Context, cands: list[dict], plan_bags: int) -> list[dict] | None:
    time_for_llm = ctx.seconds_left - 8          # keep 8 s to send the batch before the round closes
    if time_for_llm < 5:
        ctx.notes.append("no time for the LLM this round")
        return None
    user, shown = llm_prompt(state, ctx, cands, plan_bags)
    if user is None:
        ctx.notes.append("llm skipped: prompt does not fit the context")
        return None
    ctx.notes.append(f"llm prompt ~{est_tokens(SYSTEM) + est_tokens(user)} tok, {len(shown)} candidates")
    answer = ctx.llm.ask_json(SYSTEM, user, SCHEMA, timeout=min(time_for_llm, 90))
    if answer is None or not isinstance(answer.get("offers"), list):
        ctx.notes.append(f"llm failed: {getattr(ctx.llm, 'last_error', '?')}")
        return None
    plan = []
    for o in answer["offers"]:
        try:
            c = shown[int(o["c"])]
            k = float(o["k"])
            if not (c["k_lo"] - 0.001 <= k <= c["k_hi"] + 0.001):
                ctx.notes.append(f"llm k {k} outside {c['k_lo']}-{c['k_hi']} for {c['rfq_id']}, clamped")
            plan.append({"lot_id": c["lot_id"], "rfq_id": c["rfq_id"], "bags": int(o["bags"]),
                         "markup": min(max(k, c["k_lo"]), c["k_hi"])})
        except (KeyError, IndexError, TypeError, ValueError):
            ctx.notes.append(f"llm bad offer ignored: {o}")
    ctx.notes.append(f"llm reason: {str(answer.get('reason'))[:200]}")
    return plan


EDIT_SYSTEM = """You are the selling agent of a coffee farm in a sealed-offer market with ~56 competing sellers.
k = price / true value. Each buyer drops offers above its hidden ceiling (ABOVE_LIMIT), then buys the lowest k
first until full (OUTBID = others were cheaper). Unsold bags are worth only k=0.80 at the end.
You receive this round's plan from a pricing model that learnt each buyer's ceiling. You decide what is sent:
keep a row, drop it, change its bags (up to max_bags), or move its k inside k_lo..k_hi. An empty list holds stock.
Read the evidence: offers/wanted > 1.3 = crowded (others bid low, lower k); < 1.0 or bought < wanted = scarce
(you can ask more). ABOVE_LIMIT = that buyer's ceiling is lower; ACCEPTED every time = ask a little more.
1-bag rows with a higher k are probes that learn a ceiling: keep them in early rounds.
""" + EVIDENCE + "\nAnswer only JSON matching the schema."
EDIT_SCHEMA = {"type": "object", "required": ["offers", "reason"], "properties": {
    "offers": {"type": "array", "items": {"type": "object", "required": ["i", "bags", "k"], "properties": {
        "i": {"type": "integer"}, "bags": {"type": "integer"}, "k": {"type": "number"}}}},
    "reason": {"type": "string", "maxLength": 200}}}


def llm_edit(state: dict, ctx: Context, fallback: list[dict], cands: list[dict]) -> list[dict] | None:
    """LLM_MODE=editor: the LLM decides starting from the planner's plan (slide 11: propose k, sell or hold)."""
    time_for_llm = ctx.seconds_left - 8
    if time_for_llm < 5 or not fallback:
        ctx.notes.append("no time for the LLM this round" if fallback else "planner offers nothing")
        return None
    by_pair = {(c["lot_id"], c["rfq_id"]): c for c in cands}
    rows = [(p, by_pair[(p["lot_id"], p["rfq_id"])]) for p in fallback if (p["lot_id"], p["rfq_id"]) in by_pair]
    table = [[i, c["rfq_id"], c["buyer"], c["lot_id"][-5:], p["bags"], c["max_bags"], round(p["markup"], 3),
              p.get("p"), c["k_lo"], c["k_hi"]] for i, (p, c) in enumerate(rows)]
    user = None
    for whole, stats in ((True, True), (False, True), (False, False)):     # drop history until it fits the context
        text = ascii_text("\n".join(filter(None, [
            round_header(state, ctx),
            market_notes(),
            f"Calendar: {calendar_ahead(state, ctx)}.",
            "Position:", position_text(state, ctx),
            *(["Whole game per buyer:", game_summary(ctx)] if whole else []),
            *(["Market per buyer (last rounds):", market_stats(ctx)] if stats else []),
            "My offers (last rounds):", my_results(ctx),
            "Plan [i, rfq, buyer, lot, bags, max_bags, k, p_sell, k_lo, k_hi]:",
            json.dumps(table, separators=(",", ":")),
            "Return the offers to send (rows you keep, with bags and k). An empty list = hold everything this round.",
        ])))
        if fits(EDIT_SYSTEM, text):
            user = text
            break
    if user is None:
        ctx.notes.append("llm skipped: prompt does not fit the context")
        return None
    ctx.notes.append(f"llm prompt ~{est_tokens(EDIT_SYSTEM) + est_tokens(user)} tok, {len(rows)} plan rows")
    answer = ctx.llm.ask_json(EDIT_SYSTEM, user, EDIT_SCHEMA, timeout=min(time_for_llm, 90))
    if answer is None or not isinstance(answer.get("offers"), list):
        ctx.notes.append(f"llm failed: {getattr(ctx.llm, 'last_error', '?')}")
        return None
    plan = []
    for o in answer["offers"]:
        try:
            p, c = rows[int(o["i"])]
            k = float(o["k"])
            if not (c["k_lo"] - 0.001 <= k <= c["k_hi"] + 0.001):
                ctx.notes.append(f"llm k {k} outside {c['k_lo']}-{c['k_hi']} for {c['rfq_id']}, clamped")
            plan.append({**p, "bags": min(int(o["bags"]), c["max_bags"]), "markup": min(max(k, c["k_lo"]), c["k_hi"])})
        except (KeyError, IndexError, TypeError, ValueError):
            ctx.notes.append(f"llm bad offer ignored: {o}")
    ctx.notes.append(f"llm reason: {str(answer.get('reason'))[:200]}")
    return plan


def guard(state: dict, ctx: Context, plan: list[dict]) -> list[dict]:
    """Turn a plan into offers the market accepts. Never trust the LLM's numbers as they are."""
    lots = {l["lot_id"]: l for l in state["lots"]}
    rfqs = {r["rfq_id"]: r for r in state["rfqs"]}
    learner = getattr(ctx, "_learner", None)
    room = state["capacity_left_kg"] if state.get("capacity_left_kg") is not None else 10**9
    lot_left = {i: l["kg_available"] for i, l in lots.items()}
    rfq_left = {i: r["kg"] for i, r in rfqs.items()}
    offers, seen = [], set()
    # pattern spread also binds the LLM: live 12:38 the editor merged pattern's rows into one 5-bag offer on one RFQ
    # (OUTBID twice -> nothing sold). Not in the last round (everything left must be offered).
    per_offer = 10**9
    if (os.environ.get("PATTERN", "0") == "1" and state.get("capacity_left_kg") is not None
            and rounds_left(state, ctx) > 1):
        cap_bags = state["capacity_left_kg"] // BAG
        per_offer = max(1, math.ceil(cap_bags * float(os.environ.get("PATTERN_SPREAD", "0.5")))) * BAG
    # mid = off (default): arena 12 games Day 18 - off 0.9611, near-step lots q25 0.9557, lo 0.9501. Turn on
    # (TV_MARGIN=q25) only if the live market shows ABOVE_LIMIT on near-step lots.
    margin_ref = os.environ.get("TV_MARGIN", "mid")      # lo | q25 | mid
    try:
        bands = position_facts(state, ctx)["bands"] if margin_ref != "mid" else {}
    except Exception as e:                               # noqa: BLE001 - no band -> no margin, still sell
        ctx.notes.append(f"position failed: {e!r}")
        bands = {}
    try:
        bans = spec_bans(ctx)
    except Exception as e:                               # noqa: BLE001 - no bans, still sell
        ctx.notes.append(f"spec bans failed: {e!r}")
        bans = set()
    for p in plan:
        lot, rfq = lots.get(p["lot_id"]), rfqs.get(p["rfq_id"])
        if lot is None or rfq is None or not spec_ok(lot["sample"], rfq["spec"], ctx.rules):
            ctx.notes.append(f"guard dropped {p}")
            continue
        if (lot["lot_id"], rfq.get("buyer")) in bans:
            ctx.notes.append(f"guard dropped {p['lot_id']}->{p['rfq_id']}: SPEC_MISMATCH seen for this buyer")
            continue
        kg = min(int(p["bags"]) * BAG, lot_left[lot["lot_id"]], rfq_left[rfq["rfq_id"]], room, per_offer) // BAG * BAG
        if kg < BAG:
            continue
        hi = MAX_MARKUP
        if learner is not None:
            hi = min(hi, learner.belief(rfq["buyer"], rfq["buyer_type"]).hi)
        # end of game: anything above salvage beats keeping it -> pattern may go down to PATTERN_LASTFLOOR in the last
        # round, and one round earlier only on crowded Instant buyers (review 13:20: a round-7 Specialty row from the
        # LLM could otherwise go out at 0.82)
        lo_k = MIN_MARKUP
        left = rounds_left(state, ctx)
        if os.environ.get("PATTERN", "0") == "1" and (left <= 1 or (left == 2 and norm_type(rfq["buyer_type"]) == "INSTANT")):
            lo_k = max(mc.SALVAGE + 0.005, min(MIN_MARKUP, float(os.environ.get("PATTERN_LASTFLOOR", "0.81"))))
        if getattr(ctx, "_k_floor", None):         # TACTIC=1 rescue / last round: down to salvage + 0.02
            lo_k = min(lo_k, ctx._k_floor)
        lot_floor = (getattr(ctx, "_lot_floor", None) or {}).get(lot["lot_id"])
        if lot_floor:                              # TACTIC=1: lot whose TV we overestimate (ABOVE while others sold)
            lo_k = min(lo_k, lot_floor)
        markup = min(max(float(p["markup"]), lo_k), max(hi, MIN_MARKUP))
        # TV error (Day 18 noise): the buyer divides by the TRUE TV, so shade k by the lot's TV band
        # only lots whose band straddles a grade step / the moisture cliff: shading every lot cost PR in the
        # arena (12 games Day 18: off 0.9611, all lots q25 0.9544, lo 0.9481)
        if lot["lot_id"] in bands and bands[lot["lot_id"]]["near_step"]:
            markup = max(markup * pos.k_margin(bands[lot["lot_id"]], margin_ref), mc.SALVAGE + 0.02)
        price = mc.price_for(markup, ctx.tv[lot["lot_id"]])
        if not 0 < price <= 1000 or (p["lot_id"], p["rfq_id"], price) in seen:
            continue
        offers.append({"rfq_id": rfq["rfq_id"], "lot_id": lot["lot_id"], "kg": kg, "price_per_kg": price})
        seen.add((p["lot_id"], p["rfq_id"], price))
        lot_left[lot["lot_id"]] -= kg
        rfq_left[rfq["rfq_id"]] -= kg
        room -= kg
    return offers


def decide(state: dict, ctx: Context) -> tuple[list[dict], str]:
    """Returns (offers, source): "llm", "rules+llm (a+b)" (last round), "rules" or "none"."""
    try:                                     # position is advice + logging: never let it stop the round
        ctx.notes.append("position: " + position_text(state, ctx, for_log=True).replace("\n", " | "))
    except Exception as e:                   # noqa: BLE001
        ctx.notes.append(f"position failed: {e!r}")
    fallback = rule_plan(state, ctx)
    cands = candidates(state, ctx)
    if not cands:
        return [], "none"
    if os.environ.get("STRATEGY", "") == "adaptive":       # copy of the live winner (+ tactic on top if TACTIC=1)
        rows = adaptive_plan(state, ctx)
        if os.environ.get("TACTIC", "0") == "1":
            return _decide_tactic(state, ctx, rows, cands)
        return guard(state, ctx, rows), "adaptive"
    if os.environ.get("TACTIC", "0") == "1":
        return _decide_tactic(state, ctx, fallback, cands)
    if ctx.llm is not None and os.environ.get("USE_LLM", "1") == "1":
        if os.environ.get("LLM_MODE", "editor") == "seller":
            plan = llm_plan(state, ctx, cands, sum(p["bags"] for p in fallback))
        else:
            plan = llm_edit(state, ctx, fallback, cands)
        if plan is not None:
            if rounds_left(state, ctx) <= 1 and state["round"]["kind"] != "practice":
                # last round: the planner's sell-everything offers go first, the LLM's extras use what is left
                offers = guard(state, ctx, fallback + plan)
                n_rules = len(guard(state, ctx, fallback))
                return offers, f"rules+llm ({n_rules}+{len(offers) - n_rules})"
            offers = guard(state, ctx, plan)
            # live 18 Nov: the editor held everything (0 kg) in a round where stock was already behind. Holding is
            # fine while the remaining rounds can still place the stock; when they cannot (stock > capacity x rounds
            # left x 0.6) and the LLM offers less than LLM_MIN_SHARE of the planner's bags, the planner tops it up.
            share = float(os.environ.get("LLM_MIN_SHARE", "0.5"))
            want = sum(int(p["bags"]) for p in fallback) * BAG
            stock = sum(l["kg_available"] for l in state["lots"])
            cap = state.get("capacity_left_kg")
            behind = cap is not None and stock > cap * rounds_left(state, ctx) * 0.6
            # live 12:28-12:36: the editor held everything in R6 of both games -> no holding in the last N rounds
            late = rounds_left(state, ctx) <= int(os.environ.get("LLM_LATE_TOPUP", "3"))
            if share > 0 and (behind or late) and want and sum(o["kg"] for o in offers) < share * want:
                topped = guard(state, ctx, plan + fallback)
                ctx.notes.append(f"llm offered {sum(o['kg'] for o in offers)} kg < {share:.0%} of planner {want} kg"
                                 f" -> topped up to {sum(o['kg'] for o in topped)} kg")
                return topped, f"llm+rules ({len(offers)}+{len(topped) - len(offers)})"
            return offers, "llm"
    return guard(state, ctx, fallback), "rules"


def _decide_tactic(state: dict, ctx: Context, fallback: list[dict], cands: list[dict]) -> tuple[list[dict], str]:
    """TACTIC=1: the plan (LLM editor or planner) goes through tactic.apply (feedback, mode, LLM clamp, spread)."""
    import tactic
    practice = state["round"]["kind"] == "practice"
    left = rounds_left(state, ctx)
    plan, source = None, "rules"
    if ctx.llm is not None and os.environ.get("USE_LLM", "1") == "1":
        plan = llm_edit(state, ctx, fallback, cands)
        if plan is not None:
            source = "llm"
    try:
        rows = tactic.apply(state, ctx, plan if plan is not None else fallback, fallback, left, practice)
    except Exception as e:                   # noqa: BLE001 - the plain plan still sells
        ctx.notes.append(f"tactic failed, plan sent as it was: {e!r}")
        rows = plan if plan is not None else fallback
    offers = guard(state, ctx, rows)
    if left <= 1 and not practice:           # last round: whatever capacity is left goes at the planner's rows
        offers = guard(state, ctx, rows + fallback)
    if os.environ.get("FREE_TRY", "1") == "1":
        try:
            offers = offers + free_tries(state, ctx, offers)
        except Exception as e:               # noqa: BLE001 - extra tries only
            ctx.notes.append(f"free try skipped: {e!r}")
    return offers, source + "+tactic"


def free_tries(state: dict, ctx: Context, offers: list[dict]) -> list[dict]:
    """Capacity the round would leave idle goes to lots whose SAMPLE just misses every spec (pass chance 5 % up to SPEC_P): the
    buyer checks the TRUE quality, a miss costs nothing but that idle slot (SPEC_MISMATCH), a hit sells a lot nobody
    else offers (mock20 19 Nov: sample moisture 15.5, true 14.89 passes B6). Never twice to a buyer that refused it."""
    room = (state["capacity_left_kg"] if state.get("capacity_left_kg") is not None else 10 ** 9) - sum(o["kg"] for o in offers)
    if room < BAG or state["round"]["kind"] == "practice":
        return []
    used = {o["lot_id"] for o in offers}
    rfq_left = {r["rfq_id"]: r["kg"] for r in state["rfqs"]}
    for o in offers:
        rfq_left[o["rfq_id"]] = rfq_left.get(o["rfq_id"], 0) - o["kg"]
    bans = spec_bans(ctx)
    noise = _noise(ctx.rules) or {"defect": 0.35, "moisture": 0.30}
    kmap = {"SPECIALTY": 1.05, "COMMERCIAL": 0.97, "INSTANT": 0.88}
    top = float(os.environ.get("SPEC_P", "0.5"))     # at or above SPEC_P the lot is a normal candidate: the plan's job
    extra = []
    for lot in sorted(state["lots"], key=lambda l: -ctx.tv.get(l["lot_id"], 0)):
        if lot["lot_id"] in used or lot["kg_available"] < BAG or room < BAG:
            continue
        best = None
        for r in state["rfqs"]:
            pp = mc.pass_probability(lot["sample"], r["spec"], noise)
            if 0.05 <= pp < top and rfq_left.get(r["rfq_id"], 0) >= BAG and (lot["lot_id"], r.get("buyer")) not in bans:
                if best is None or pp > best[0]:
                    best = (pp, r)
        if best is None:
            continue
        r = best[1]
        kg = min(lot["kg_available"], rfq_left[r["rfq_id"]], room) // BAG * BAG
        price = mc.price_for(kmap.get(norm_type(r["buyer_type"]), 0.9), ctx.tv[lot["lot_id"]])
        if kg >= BAG and 0 < price <= 1000:
            extra.append({"rfq_id": r["rfq_id"], "lot_id": lot["lot_id"], "kg": kg, "price_per_kg": price})
            room -= kg
            rfq_left[r["rfq_id"]] -= kg
            ctx.notes.append(f"free try {lot['lot_id'][-5:]} -> {r['rfq_id']} (sample passes {best[0]:.0%}), {kg // BAG} bags")
    return extra


# ---------------------------------------------------------------------- STRATEGY=adaptive (copy of the live winner)
# The live practice room's "บอตปรับราคา" (adaptive bot) finished first by PR in 27 of 32 recorded live games and held
# 131 of 160 top-5 places (research/live-market/games). Its rule (tesareplica/bots.py, fitted on those games):
# one k per buyer TYPE, start SPECIALTY 1.12 / COMMERCIAL 1.00 / INSTANT 0.90, +0.01 after a round where that type
# bought something from us, -0.01 after a round where it bought nothing; Specialty priced like Commercial after R3 if
# it never bought; every lot offers 1.5 x its even share of the rounds left, best lots first, Specialty RFQs first.
ADAPT_START = {"SPECIALTY": 1.12, "COMMERCIAL": 1.00, "INSTANT": 0.90}


def adaptive_k(ctx: Context) -> tuple[dict, bool]:
    step = float(os.environ.get("ADAPT_STEP", "0.01"))
    k = {t: float(os.environ.get(f"ADAPT_{t[:3]}", v)) for t, v in ADAPT_START.items()}
    spe_sold = False
    for rnd in (ctx.memory.rounds if ctx.memory else []):
        kinds = {m["rfq_id"]: norm_type(m["buyer_type"]) for m in rnd.get("market_summary", [])}
        res: dict = {}
        for o in rnd.get("my_offers", []):
            t = kinds.get(o["rfq_id"])
            if t:
                res.setdefault(t, set()).add(o["status"])
        for t, sts in res.items():
            if sts & {"ACCEPTED", "PARTIAL"}:
                k[t] += step
                spe_sold = spe_sold or t == "SPECIALTY"
            elif sts & {"OUTBID", "ABOVE_LIMIT"}:
                k[t] -= step
            k[t] = round(min(MAX_MARKUP, max(MIN_MARKUP, k[t])), 3)
    return k, spe_sold


def adaptive_plan(state: dict, ctx: Context) -> list[dict]:
    k, spe_sold = adaptive_k(ctx)
    number, _total = _round_numbers(state, ctx)
    if number > 3 and not spe_sold:
        k["SPECIALTY"] = min(k["SPECIALTY"], k["COMMERCIAL"])
    left = max(1, rounds_left(state, ctx))
    room = (state["capacity_left_kg"] if state.get("capacity_left_kg") is not None else 10 ** 9) // BAG
    demand = {r["rfq_id"]: r["kg"] // BAG for r in state["rfqs"]}
    tier = {"SPECIALTY": 0, "COMMERCIAL": 1, "INSTANT": 2}
    draw = {r["rfq_id"]: ctx.rng.random() ** (1 / max(1, r["bags"])) for r in state["rfqs"]}
    order = sorted(state["rfqs"], key=lambda r: (tier.get(norm_type(r["buyer_type"]), 3), -draw[r["rfq_id"]]))
    pace_x = float(os.environ.get("ADAPT_PACE", "1.5"))
    try:
        bans = spec_bans(ctx)
    except Exception:                                # noqa: BLE001
        bans = set()
    rows = []
    for lot in sorted(state["lots"], key=lambda l: -ctx.tv.get(l["lot_id"], 0)):
        avail = lot["kg_available"] // BAG
        pace = min(avail, max(1, math.ceil(pace_x * avail / left)))
        if left <= 1:
            pace = avail
        for r in order:
            if pace <= 0 or room <= 0:
                break
            if not spec_ok(lot["sample"], r["spec"], ctx.rules) or (lot["lot_id"], r.get("buyer")) in bans:
                continue
            bags = min(pace, room, demand[r["rfq_id"]])
            if bags <= 0:
                continue
            rows.append({"lot_id": lot["lot_id"], "rfq_id": r["rfq_id"], "bags": bags,
                         "markup": k[norm_type(r["buyer_type"])], "p": None})
            pace -= bags
            room -= bags
            demand[r["rfq_id"]] -= bags
    ctx.notes.append(f"adaptive k {k}")
    return rows
