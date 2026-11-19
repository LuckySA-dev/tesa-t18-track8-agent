"""lpalloc.assign: horizon transportation LP (min-cost flow) - stock to the buyers where it is worth most."""
import os
import random

import lpalloc
import pattern
from pricing import CeilingLearner


def test_best_lot_goes_to_scarce_premium_buyer_rest_elsewhere():
    stock = {"TOP": 10, "MID": 10}
    demand = {"SPE": 4, "COM": 20}
    value = {("TOP", "SPE"): 50, ("TOP", "COM"): 30, ("MID", "COM"): 25}
    a = lpalloc.assign(stock, demand, value)
    assert a["TOP"]["SPE"] == 4                         # the premium buyer's small demand goes to the lot that fits it
    assert a["TOP"]["COM"] + a["MID"]["COM"] == 16     # all the other 16 bags go to Commercial
    assert sum(a["TOP"].values()) <= 10


def test_exchange_argument_beats_greedy():
    # greedy by value would put A at X (60) and leave B unplaced; the optimum is A->Y, B->X
    stock = {"A": 1, "B": 1}
    demand = {"X": 1, "Y": 1}
    value = {("A", "X"): 60, ("A", "Y"): 55, ("B", "X"): 50}
    a = lpalloc.assign(stock, demand, value)
    assert a == {"A": {"Y": 1}, "B": {"X": 1}}


def test_total_capacity_and_nonpositive_values():
    a = lpalloc.assign({"A": 10}, {"X": 10, "Y": 10}, {("A", "X"): 5, ("A", "Y"): -1}, total_cap=3)
    assert a == {"A": {"X": 3}}


def test_horizon_demand_uses_history_share():
    rounds = [{"round_id": "R1", "market_summary": [
        {"buyer": "S", "wanted_kg": 180, "offers": 1}, {"buyer": "I", "wanted_kg": 1800, "offers": 20}]}]
    buyers = {"S": ("SPECIALTY", {}), "I": ("INSTANT", {})}
    d = pattern.horizon_demand(rounds, [], buyers, left_rounds=4, share_k=2.0)
    assert d["S"] == round(3 * 0.6 * 4)                 # few rivals: capped share 0.6
    assert d["I"] == round(30 * 0.1 * 4)                # 20 offers per RFQ: 2/20 of it


def test_pattern_lp_mode_runs_and_respects_capacity(monkeypatch):
    monkeypatch.setenv("PATTERN_ALLOC", "lp")
    spe = {"max_defect_pct": 3.0, "max_moisture_pct": 12.5, "min_screen": 16}
    ins = {"max_defect_pct": 15.0, "max_moisture_pct": 15.0, "min_screen": None}
    rfqs = [{"rfq_id": "R2-01", "buyer": "1 S", "buyer_type": "SPECIALTY", "spec": spe, "bags": 3, "kg": 180},
            {"rfq_id": "R2-02", "buyer": "6 I", "buyer_type": "INSTANT", "spec": ins, "bags": 20, "kg": 1200}]
    lots = [{"lot_id": "TOP", "kg_available": 600, "sample": {"defect_pct": 1.0, "moisture_pct": 11.0, "screen": 17}},
            {"lot_id": "LOW", "kg_available": 600, "sample": {"defect_pct": 10.0, "moisture_pct": 13.0, "screen": 14}}]
    st = {"round": {"kind": "real", "number": 2}, "lots": lots, "rfqs": rfqs, "capacity_left_kg": 300}
    notes = []
    rows = pattern.plan(st, {"TOP": 130.0, "LOW": 100.0}, CeilingLearner(), [], None, {"timeline": {"rounds": 8}},
                        None, random.Random(0), notes)
    assert sum(r["bags"] for r in rows) <= 5
    assert any(n.startswith("pattern lp:") for n in notes)
    assert os.environ["PATTERN_ALLOC"] == "lp"
