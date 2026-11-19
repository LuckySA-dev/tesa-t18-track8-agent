"""pattern.plan: lots go to the buyers they fit best; gap buyers are priced from what they accepted, not k*."""
import random

import pattern
from pricing import CeilingLearner

SPE = {"max_defect_pct": 3.0, "max_moisture_pct": 12.5, "min_screen": 16}
COM = {"max_defect_pct": 7.0, "max_moisture_pct": 12.5, "min_screen": None}
INS = {"max_defect_pct": 15.0, "max_moisture_pct": 15.0, "min_screen": None}
RULES = {"timeline": {"rounds": 8}}


def rfq(i, name, t, spec, bags=10):
    return {"rfq_id": f"R2-0{i}", "buyer": name, "buyer_type": t, "spec": spec, "bags": bags, "kg": bags * 60}


def state(number, rfqs, cap=600):
    lots = [{"lot_id": "TOP", "kg_available": 300, "sample": {"defect_pct": 1.0, "moisture_pct": 11.0, "screen": 17}},
            {"lot_id": "LOW", "kg_available": 600, "sample": {"defect_pct": 10.0, "moisture_pct": 13.0, "screen": 14}}]
    return {"round": {"kind": "real", "number": number}, "lots": lots, "rfqs": rfqs, "capacity_left_kg": cap}


def run(st, rounds=(), known=None):
    notes = []
    rows = pattern.plan(st, {"TOP": 130.0, "LOW": 100.0}, CeilingLearner(), list(rounds), known, RULES, None,
                        random.Random(0), notes)
    return rows, notes


ALL = [rfq(1, "1 SPE", "SPECIALTY", SPE, 3), rfq(2, "3 COM", "COMMERCIAL", COM), rfq(3, "6 INS", "INSTANT", INS)]


def test_top_lot_goes_only_to_its_best_fit_early():
    rows, _ = run(state(2, ALL))
    top = {r["rfq_id"] for r in rows if r["lot_id"] == "TOP"}
    assert top == {"R2-01"}                                  # Specialty only, not Commercial / Instant
    assert {r["rfq_id"] for r in rows if r["lot_id"] == "LOW"} == {"R2-03"}   # LOW only fits Instant


def test_top_lot_waits_when_its_buyer_is_absent_early_but_sells_late():
    no_spe = ALL[1:]
    known = {"1 SPE": {"type": "SPECIALTY", "spec": SPE}}
    early, _ = run(state(2, no_spe), known=known)
    assert not [r for r in early if r["lot_id"] == "TOP"]    # keep it for the Specialty buyer
    late, _ = run(state(7, no_spe), known=known)
    assert [r for r in late if r["lot_id"] == "TOP"]         # last rounds: every fitting buyer opens


def test_gap_buyer_climbs_while_never_cut():
    lr = CeilingLearner()
    ev = lr._ev("1 SPE", "SPECIALTY")[0]
    ev.passed.append(1.10)
    k, _ = pattern.buyer_k("1 SPE", "SPECIALTY", "scarce", lr, 0.85, 0.01, 0.005)
    assert abs(k - 1.11) < 1e-9
    ev.above.append(1.15)
    k, _ = pattern.buyer_k("1 SPE", "SPECIALTY", "scarce", lr, 0.85, 0.01, 0.005)
    assert abs(k - 1.095) < 1e-9                             # ceiling known: just under what it accepted


def test_no_single_offer_takes_the_round_and_low_lot_gets_slots():
    # live game 18 Nov 11:41: 7-bag capacity went to ONE offer of the best lot; the other lots were never offered
    st = state(2, ALL, cap=420)
    st["lots"][0]["kg_available"] = 1200                     # big TOP lot
    rows, _ = run(st, known=None)
    assert max(r["bags"] for r in rows) <= 4                 # <= half the round's 7 bags (rounded up)
    assert any(r["lot_id"] == "LOW" for r in rows)           # the Instant-only lot still gets its share


def test_respects_capacity_and_floor():
    rows, _ = run(state(2, ALL, cap=120))
    assert sum(r["bags"] for r in rows) <= 2
    assert all(r["markup"] >= 0.85 for r in rows)
