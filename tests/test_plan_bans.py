"""Banned (lot, buyer) pairs must not take capacity in the planner (replica 19 Nov regression).

A Specialty-grade lot that returned SPEC_MISMATCH at B1/B2 kept winning the greedy allocation, the guard then
dropped those offers, and the round went out empty while Instant-only lots never got a bag of capacity.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planner as pl                                   # noqa: E402
from pricing import CeilingLearner                     # noqa: E402

SPE = {"max_defect_pct": 3.0, "max_moisture_pct": 12.5, "min_screen": 16}
INS = {"max_defect_pct": 15.0, "max_moisture_pct": 15.0, "min_screen": None}
LOTS = [{"lot_id": "GOOD", "kg_available": 600, "sample": {"defect_pct": 2.0, "moisture_pct": 11.0, "screen": 17}},
        {"lot_id": "WET", "kg_available": 600, "sample": {"defect_pct": 4.0, "moisture_pct": 13.5, "screen": 15}}]
RFQS = [{"rfq_id": "R3-01", "buyer": "1 SPE", "buyer_type": "SPECIALTY", "kg": 360, "bags": 6, "spec": SPE},
        {"rfq_id": "R3-02", "buyer": "6 INS", "buyer_type": "INSTANT", "kg": 1200, "bags": 20, "spec": INS}]
TV = {"GOOD": 125.0, "WET": 110.0}


def plan(bans):
    return pl.plan_round(LOTS, RFQS, TV, CeilingLearner(), round_number=3, rounds_total=8, capacity_left_kg=360,
                         bans=bans)


def test_without_bans_specialty_lot_takes_the_room():
    offers = plan(None).offers
    assert any(o.lot_id == "GOOD" and o.rfq_id == "R3-01" for o in offers)


def test_banned_pair_is_skipped_and_room_goes_to_other_lots():
    p = plan({("GOOD", "1 SPE")})
    assert not any(o.lot_id == "GOOD" and o.rfq_id == "R3-01" for o in p.offers)
    assert any(r == "R3-01" and "SPEC_MISMATCH" in why for _, r, why in p.skipped)
    assert sum(o.bags for o in p.offers) == 6          # the whole capacity is still used
    assert any(o.lot_id == "WET" for o in p.offers)
