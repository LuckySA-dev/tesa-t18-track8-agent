"""Where we stand in the market (Track 3-2 engine, pure maths, no I/O).

Facts this module is built on (18 Nov):
  * Every team gets the same NUMBER of bags but different lots, so TV and the buyer types a lot can reach differ
    per team. Buyers (specs, ceilings) and the TV formula are the same for everyone.
  * Samples are noisy on Day 18: the buyer ranks by price / TRUE TV and checks the spec on TRUE quality, so every
    TV we use is an estimate with an error band - widest near a grade step or near moisture 15 %.
  * Specs per buyer are fixed; what moves between rounds is how many bags each buyer wants (calendar
    demand_weight / specialty_share) and how many sellers still have stock.

Nothing here is hard-coded to one market: noise, demand, team count and calendar weights are inputs read from
the API every round. Default noise is only a starting guess until reveal data replaces it.

    lot_band(sample, rules, noise, specs)      TV low / mid / high, relative error, P(spec ok) per buyer
    pot_band(lots, bands, revenue)             pot low / mid / high, PR floor, PR if the rest sells at k
    calendar_weights(calendar, rounds)         {round: (demand_weight, specialty_share)}
    market_balance(rounds, n_teams, bags, ...) demand vs supply per buyer type, now and projected
    buyer_gaps(rounds)                         per buyer: ceiling-bound / scarce / crowded / balanced
    allocation_pull(lots, bands, balance)      which buyer type each lot should go to, which to hold
    k_margin(band, q)                          scale k so the TRUE k stays under target with prob ~(1 - q)
"""
from __future__ import annotations

import re
import statistics

import market_core as mc

BAG = mc.BAG_KG
TYPES = ("SPECIALTY", "COMMERCIAL", "INSTANT")
DEFAULT_NOISE = {"defect": 0.35, "moisture": 0.30}     # organiser engine guess - replace with reveal data


# ---------------------------------------------------------------------- 1. lot value band

def _tv_grid(sample: dict, rules: dict | None, noise: dict | None, steps: int = 21) -> list[float]:
    nd, nm = (noise or {}).get("defect", 0), (noise or {}).get("moisture", 0)
    out = []
    for i in range(steps):
        d = max(0.0, sample["defect_pct"] + nd * (2 * i / (steps - 1) - 1)) if nd else sample["defect_pct"]
        for j in range(steps):
            m = max(0.0, sample["moisture_pct"] + nm * (2 * j / (steps - 1) - 1)) if nm else sample["moisture_pct"]
            out.append(mc.tv({"defect_pct": d, "moisture_pct": m, "screen": sample["screen"]}, rules))
            if not nm:
                break
        if not nd:
            break
    return sorted(out)


def lot_band(sample: dict, rules: dict | None = None, noise: dict | None = None,
             specs: dict | None = None) -> dict:
    """TV band of one lot under sample noise. specs = {buyer code: spec} -> P(true quality passes) each."""
    grid = _tv_grid(sample, rules, noise)
    lo, hi = grid[0], grid[-1]
    mid = mc.appraisal_estimate(sample, rules, noise) if noise else round(grid[0], 2)
    q = lambda p: grid[min(len(grid) - 1, int(p * len(grid)))]                       # noqa: E731
    return {"lo": round(lo, 2), "q25": round(q(0.25), 2), "mid": mid, "hi": round(hi, 2),
            "err": round(max(mid - lo, hi - mid) / mid, 4) if mid else 0.0,
            # a step inside the band = the lot may really be one grade (or the 0.60 moisture cliff) away
            "near_step": (hi - lo) / mid > 0.02 if mid else False,
            "p_spec": {code: round(mc.pass_probability(sample, spec, noise), 3) for code, spec in (specs or {}).items()}}


# ---------------------------------------------------------------------- 2. pot

def pot_band(lots: list[dict], bands: dict, revenue: float = 0.0, k_rest: float = 1.0) -> dict:
    """Our pot (TV x kg of everything we got) and where the score can land from here.

    pr_floor: every bag still unsold goes to salvage (0.80). pr_at_k: the rest sells at k_rest.
    Both use the mid TV; *_lo / *_hi show how far TV error alone can move the denominator."""
    pot = {key: sum(bands[l["lot_id"]][key] * l["kg_total"] for l in lots) for key in ("lo", "mid", "hi")}
    left = sum(bands[l["lot_id"]]["mid"] * l["kg_remaining"] for l in lots)
    sold_value = pot["mid"] - left
    out = {"pot_lo": round(pot["lo"]), "pot_mid": round(pot["mid"]), "pot_hi": round(pot["hi"]),
           "bags_total": sum(l["kg_total"] for l in lots) // BAG, "bags_left": sum(l["kg_remaining"] for l in lots) // BAG,
           "revenue": round(revenue)}
    if pot["mid"]:
        out["pr_so_far"] = round(revenue / sold_value, 4) if sold_value > 0 else None
        out["pr_floor"] = round((revenue + mc.SALVAGE * left) / pot["mid"], 4)
        out["pr_at_k"] = round((revenue + k_rest * left) / pot["mid"], 4)
    return out


# ---------------------------------------------------------------------- 3. calendar + market balance

def calendar_weights(calendar: list | None, rounds: int) -> dict[int, tuple[float, float | None]]:
    """{round: (demand_weight, specialty_share)} from GET /v1/calendar rows. Missing fields -> (1.0, None)."""
    out = {}
    for row in calendar or []:
        try:
            if "rounds" in row:
                nums = [int(x) for x in re.findall(r"\d+", str(row["rounds"]))]
                lo, hi = nums[0], nums[-1]
            else:
                lo, hi = int(row.get("from", 0)), int(row.get("to", 0))
            w = float(row.get("demand_weight") or 1.0)
            share = row.get("specialty_share")
            share = float(share) if share is not None else None
        except (TypeError, ValueError, IndexError, AttributeError):
            continue
        for n in range(lo, hi + 1):
            out[n] = (w, share)
    return {n: out.get(n, (1.0, None)) for n in range(1, rounds + 1)}


def _type(row: dict) -> str:
    t = (row.get("buyer_type") or row.get("type") or "").upper()
    return {"C": "COMMERCIAL", "I": "INSTANT", "S": "SPECIALTY", "SPE": "SPECIALTY", "COM": "COMMERCIAL",
            "INS": "INSTANT"}.get(t, t)


def market_balance(rounds: list[dict], n_teams: int | None, bags_per_team: int, round_number: int,
                   rounds_total: int, weights: dict | None = None, rfqs_now: list[dict] | None = None) -> dict:
    """Demand vs supply for the whole market, now and over the rounds left.

    rounds: cleared scored results (market_summary rows). Each team got the same number of bags, so market supply
    at the start is n_teams x bags_per_team; every bag bought leaves it, so supply_left = start - sum(bought).
    Demand per round per type is the mean seen so far (or this round's RFQs), scaled by each future round's
    demand_weight. pressure = supply_left / demand_left: > 1 sellers outnumber bags wanted (prices fall),
    < 1 buyers will be short (prices can rise to the ceiling)."""
    weights = weights or {}
    per_type, bought, offers = {t: [] for t in TYPES}, 0, {t: [] for t in TYPES}
    for rnd in rounds:
        agg = {t: [0, 0] for t in TYPES}
        for row in rnd.get("market_summary", []):
            t = _type(row)
            if t in agg:
                agg[t][0] += row["wanted_kg"] // BAG
                agg[t][1] += row.get("offers") or 0
                bought += row["bought_kg"] // BAG
        w = weights.get(int(re.sub(r"\D", "", rnd.get("round_id", "")) or 0), (1.0, None))[0] or 1.0
        for t, (wanted, n_off) in agg.items():
            if wanted:
                per_type[t].append(wanted / w)          # demand at weight 1.0
                offers[t].append(n_off / wanted)
    now = {t: 0 for t in TYPES}
    for r in rfqs_now or []:
        if _type(r) in now:
            now[_type(r)] += r.get("bags") or r["kg"] // BAG
    base = {t: (statistics.mean(per_type[t]) if per_type[t] else now[t]) for t in TYPES}
    future = range(round_number, rounds_total + 1)
    demand_left = {t: round(sum(base[t] * weights.get(n, (1.0, None))[0] for n in future)) for t in TYPES}
    out = {"demand_per_round": {t: round(v, 1) for t, v in base.items()}, "demand_left": demand_left,
           "crowd_trend": {t: [round(x, 2) for x in v[-3:]] for t, v in offers.items() if v}}
    if n_teams:
        supply_start = n_teams * bags_per_team
        supply_left = max(0, supply_start - bought)
        total_left = sum(demand_left.values())
        out.update(n_teams=n_teams, supply_start=supply_start, supply_left=supply_left,
                   pressure=round(supply_left / total_left, 2) if total_left else None)
    return out


# ---------------------------------------------------------------------- 4. gaps per buyer

def buyer_gaps(rounds: list[dict], code_of, last: int = 2) -> dict:
    """Classify each buyer from its last rounds:
      ceiling   bought < wanted although offers >= wanted -> most bids were above its ceiling: bid just under it
      scarce    offers < wanted                            -> not enough sellers: ask up to the ceiling
      crowded   bought = wanted and offers/wanted >= 1.3   -> price war: bid under the clearing k or go elsewhere
      balanced  anything else
    """
    stats = {}
    for rnd in rounds[-last:]:
        for row in rnd.get("market_summary", []):
            s = stats.setdefault(code_of(row), [0, 0, 0])
            s[0] += row["wanted_kg"] // BAG
            s[1] += row["bought_kg"] // BAG
            s[2] += row.get("offers") or 0
    out = {}
    for code, (w, b, n) in stats.items():
        ratio = n / max(w, 1)
        kind = ("scarce" if ratio < 1 else "ceiling" if b < w else "crowded" if ratio >= 1.3 else "balanced")
        out[code] = {"kind": kind, "unfilled": w - b, "offers_per_wanted": round(ratio, 2)}
    return out


# ---------------------------------------------------------------------- 5. which lot goes where

def allocation_pull(lots: list[dict], bands: dict, codes_by_type: dict, balance: dict, p_min: float = 0.9) -> dict:
    """Per lot: buyer types it can reach (P(spec) >= p_min on any buyer of that type) and a pull score per type
    = demand left of that type / bags we could send there. A lot reaching a scarce, high-paying type is held for it
    (e.g. our best lot for Specialty); a lot that reaches only one crowded type should sell early."""
    demand = balance.get("demand_left", {})
    reach = {}
    for l in lots:
        p = bands[l["lot_id"]]["p_spec"]
        reach[l["lot_id"]] = [t for t in TYPES if any(p.get(c, 0) >= p_min for c in codes_by_type.get(t, []))]
    our_bags = {t: sum(l["kg_remaining"] // BAG for l in lots if t in reach[l["lot_id"]]) for t in TYPES}
    out = {}
    for l in lots:
        types = reach[l["lot_id"]]
        pull = {t: round(demand.get(t, 0) / max(our_bags[t], 1), 2) for t in types}
        best = max(types, key=lambda t: (TYPES.index(t) == 0, pull[t]), default=None)   # Specialty first if reachable
        out[l["lot_id"]] = {"reach": types, "pull": pull, "target": best,
                            "hold_for_specialty": best == "SPECIALTY",
                            "single_outlet": len(types) == 1}
    return out


# ---------------------------------------------------------------------- 6. what k the market actually paid

# Plausible true-quality box for any team's lot (organiser engine templates + 17 Nov lots). Only a prior: our own lots
# are mixed in, and a buyer's spec cuts the box down hard, which is what makes the back-solve usable.
QUALITY_BOX = {"defect": (0.3, 14.0), "moisture": (10.5, 14.5), "screen": (12, 18)}
_TV_CACHE: dict = {}


def eligible_tv(spec: dict, better_specs: list[dict], rules: dict | None = None, own_tvs: list[float] | None = None,
                n: int = 3000, prior_weight: int = 8) -> tuple[float, float]:
    """(mean, relative sd) of the TV of lots that would be SOLD to a buyer with `spec`: they pass it but none of the
    better-paying types' specs (a lot that also passes Commercial goes to Commercial, not Instant).
    Monte-Carlo over QUALITY_BOX, blended with our own matching lots (prior_weight = pseudo-count of the box)."""
    key = (tuple(sorted(spec.items())), tuple(tuple(sorted(s.items())) for s in better_specs), id(rules))
    if key not in _TV_CACHE:
        import random
        rng, vals = random.Random(0), []
        (d0, d1), (m0, m1), (s0, s1) = QUALITY_BOX["defect"], QUALITY_BOX["moisture"], QUALITY_BOX["screen"]
        for _ in range(n):
            q = {"defect_pct": rng.uniform(d0, d1), "moisture_pct": rng.uniform(m0, m1), "screen": rng.randint(s0, s1)}
            if mc.passes(q, spec) and not any(mc.passes(q, b) for b in better_specs):
                vals.append(mc.tv(q, rules))
        _TV_CACHE[key] = vals
    vals = _TV_CACHE[key]
    if not vals:
        return 0.0, 0.0
    box_mean = statistics.mean(vals)
    own = own_tvs or []
    mean = (box_mean * prior_weight + sum(own)) / (prior_weight + len(own))
    sd = statistics.pstdev(vals) / box_mean if len(vals) > 1 else 0.0
    return mean, sd


def implied_k(rounds: list[dict], specs: dict, code_of, type_of, rules: dict | None = None,
              own: dict | None = None) -> dict:
    """Back-solve the k other sellers were paid: k ~ bought-weighted price_avg / mean TV of the lots that buyer gets.

    specs: {buyer code: spec}; type_of(code) -> buyer type; own: {buyer code: [our lot TVs that pass it]}.
    Returns {code: {"k": [per round], "sd": relative TV uncertainty}}. Commercial is tight (its spec narrows TV a lot,
    17 Nov: two priors agreed within 0.003); Instant is loose (+-5-8 %) because its lots vary widely.
    This is the AVERAGE k of bags sold; the marginal (clearing) k is a little higher."""
    order = {"SPECIALTY": 0, "COMMERCIAL": 1, "INSTANT": 2}
    out = {}
    for code, spec in specs.items():
        better = [s for c, s in specs.items() if order.get(type_of(c), 3) < order.get(type_of(code), 3)]
        mean, sd = eligible_tv(spec, better, rules, (own or {}).get(code))
        if not mean:
            continue
        ks = []
        for rnd in rounds:
            num = den = 0.0
            for row in rnd.get("market_summary", []):
                if code_of(row) == code and row.get("price_avg") and row.get("bought_kg"):
                    num += row["price_avg"] * row["bought_kg"]
                    den += row["bought_kg"]
            if den:
                ks.append(round(num / den / mean, 3))
        if ks:
            out[code] = {"k": ks, "sd": round(sd, 3)}
    return out


# ---------------------------------------------------------------------- 7. k under TV error

def k_margin(band: dict, ref: str = "q25") -> float:
    """Factor (<= 1) to multiply k by so that price / TRUE TV stays at or under the intended k unless the true TV
    falls below band[ref]. ref = "lo" -> worst case, "q25" -> 75 % safe (default), "mid" -> no margin."""
    return min(1.0, band[ref] / band["mid"]) if band.get("mid") else 1.0
