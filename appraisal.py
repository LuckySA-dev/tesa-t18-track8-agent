"""Estimate a lot's True Value (TV) from its Sample, using the formula the market publishes.

The formula and its numbers come from GET /v1/calendar -> "rules", so nothing is hard-coded here.
"""


def grade(rules: dict, defect_pct: float) -> tuple[str, float]:
    for step in rules["defect_steps"]:
        if defect_pct <= step["max_defect_pct"]:
            return step["grade"], step["step"]
    last = rules["defect_steps"][-1]
    return last["grade"], last["step"]


def moisture_factor(rules: dict, m: float) -> float:
    mo = rules["moisture"]
    if m <= mo["ref"]:
        return 1.0
    if m <= mo["hard_limit"]:
        return (1 - m / 100) / (1 - mo["ref"] / 100) - mo["penalty_per_pp"] * (m - mo["ref"])
    return mo["hard_factor"]


def screen_factor(rules: dict, screen: int) -> float:
    factors = {int(k): v for k, v in rules["screen_factors"].items()}
    codes = sorted(factors)
    return factors[min(max(int(screen), codes[0]), codes[-1])]


def true_value(rules: dict, sample: dict) -> float:
    """TV = P0 × G(d) × M(m) × S(s), baht per kg."""
    d, m = sample["defect_pct"], sample["moisture_pct"]
    g = (1 - d / 100) * grade(rules, d)[1]
    return rules["p0"] * g * moisture_factor(rules, m) * screen_factor(rules, sample["screen"])


def estimate_lots(rules: dict, lots: list[dict]) -> dict[str, float]:
    """The sample is noisy on Day 18 (350 g only). Near a grade step the lot may really be one grade better or
    worse, so we submit the noise-aware median TV (market_core.appraisal_estimate), which minimises MAPE.
    With no noise (Day 17) this is exactly TV(sample)."""
    import market_core
    from strategy import _noise
    noise = _noise(rules)
    return {lot["lot_id"]: market_core.appraisal_estimate(lot["sample"], rules, noise) for lot in lots}


def appraisal_items(rules: dict, lots: list[dict], tv: dict[str, float]) -> list[dict]:
    return [{"lot_id": lot["lot_id"], "grade": grade(rules, lot["sample"]["defect_pct"])[0],
             "est_value_per_kg": tv[lot["lot_id"]]} for lot in lots]
