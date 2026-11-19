"""Specs are fixed per BUYER, not per type (live 18 Nov: B1 screen >= 16 vs B2 >= 17, B5 moisture <= 14 vs B6 <= 15)."""
from memory import Memory
from planner import PlanConfig, hold_value, plan_round
from pricing import CeilingLearner

B1 = {"rfq_id": "R1-01", "buyer": "1 B1", "buyer_type": "SPECIALTY", "kg": 120, "bags": 2,
      "spec": {"max_defect_pct": 3.0, "max_moisture_pct": 12.5, "min_screen": 16}}
B2 = {"rfq_id": "R1-02", "buyer": "2 B2", "buyer_type": "SPECIALTY", "kg": 120, "bags": 2,
      "spec": {"max_defect_pct": 3.0, "max_moisture_pct": 12.5, "min_screen": 17}}
B5 = {"rfq_id": "R1-03", "buyer": "5 B5", "buyer_type": "INSTANT", "kg": 600, "bags": 10,
      "spec": {"max_defect_pct": 15.0, "max_moisture_pct": 14.0, "min_screen": None}}
B6 = {"rfq_id": "R1-04", "buyer": "6 B6", "buyer_type": "INSTANT", "kg": 600, "bags": 10,
      "spec": {"max_defect_pct": 15.0, "max_moisture_pct": 15.0, "min_screen": None}}

WET = {"lot_id": "WET", "kg_available": 600, "sample": {"defect_pct": 10.0, "moisture_pct": 14.5, "screen": 14}}
S16 = {"lot_id": "S16", "kg_available": 300, "sample": {"defect_pct": 2.0, "moisture_pct": 11.0, "screen": 16}}
CFG = PlanConfig(noise=None, patience=1.0, probes=False)


def run(lots, rfqs, known=None, rn=2):
    tv = {l["lot_id"]: 100.0 for l in lots}
    return plan_round(lots, rfqs, tv, CeilingLearner(), round_number=rn, rounds_total=8, cfg=CFG, known_buyers=known)


def known(*rfqs):
    return {r["buyer"]: {"type": r["buyer_type"], "spec": r["spec"]} for r in rfqs}


def test_wet_lot_only_b6_is_not_unsellable_when_b6_is_remembered():
    assert "WET" in run([WET], [B5]).unsellable                 # only B5 (moisture <= 14) known: looks unsellable
    assert "WET" not in run([WET], [B5], known(B6)).unsellable  # B6 (<= 15) seen in an earlier round


def test_screen16_lot_keeps_specialty_hold_value_even_if_b2_comes_first():
    # old code kept the FIRST Specialty spec of the round (B2, screen >= 17) -> S16 had no Specialty value
    with_b1 = run([S16], [B2], known(B1)).reserve["S16"]
    without = run([S16], [B2]).reserve["S16"]
    assert with_b1 > without


def test_hold_value_accepts_buyer_pairs_and_plain_types():
    lr = CeilingLearner()
    assert hold_value(lr, [("1 B1", "SPECIALTY")], False) == hold_value(lr, ["SPECIALTY"], False)
    assert hold_value(lr, [], False) == 0.80


def test_memory_remembers_buyers_across_restart(tmp_path):
    m = Memory(tmp_path / "mem.json")
    assert m.remember_buyers([B1, B6]) is True
    assert m.remember_buyers([B1]) is False                     # nothing new
    m.save()
    again = Memory(tmp_path / "mem.json")
    assert again.buyers["6 B6"]["spec"]["max_moisture_pct"] == 15.0
    assert set(again.buyers) == {"1 B1", "6 B6"}
