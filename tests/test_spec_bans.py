"""A SPEC_MISMATCH bans the lot from that buyer and every buyer at least as strict (live 12:51: 7 rounds wasted)."""
import random

import strategy as st

B1 = {"max_defect_pct": 3.0, "max_moisture_pct": 12.5, "min_screen": 16}
B2 = {"max_defect_pct": 3.0, "max_moisture_pct": 12.5, "min_screen": 17}
B3 = {"max_defect_pct": 7.0, "max_moisture_pct": 12.5, "min_screen": None}


class Mem:
    buyers = {"1 B1": {"type": "SPECIALTY", "spec": B1}, "2 B2": {"type": "SPECIALTY", "spec": B2},
              "3 B3": {"type": "COMMERCIAL", "spec": B3}}
    rounds = [{"round_id": "R2", "market_summary": [{"rfq_id": "R2-01", "buyer": "1 B1"}],
               "my_offers": [{"rfq_id": "R2-01", "lot_id": "L4", "status": "SPEC_MISMATCH"}]}]


def ctx():
    rules = {"timeline": {"rounds": 8}}
    return st.Context(rules, [], {"L4": 120.0}, random.Random(0), Mem(), None, 90)


def test_ban_covers_failed_buyer_and_stricter_only():
    bans = st.spec_bans(ctx())
    assert ("L4", "1 B1") in bans and ("L4", "2 B2") in bans
    assert ("L4", "3 B3") not in bans


def test_guard_drops_banned_pair_keeps_commercial():
    state = {"round": {"kind": "real", "number": 3}, "capacity_left_kg": 600,
             "lots": [{"lot_id": "L4", "kg_available": 600,
                       "sample": {"defect_pct": 2.6, "moisture_pct": 12.0, "screen": 17}}],
             "rfqs": [{"rfq_id": "R3-01", "buyer": "1 B1", "buyer_type": "SPECIALTY", "spec": B1, "kg": 180, "bags": 3},
                      {"rfq_id": "R3-02", "buyer": "3 B3", "buyer_type": "COMMERCIAL", "spec": B3, "kg": 600, "bags": 10}]}
    plan = [{"lot_id": "L4", "rfq_id": "R3-01", "bags": 2, "markup": 1.15},
            {"lot_id": "L4", "rfq_id": "R3-02", "bags": 2, "markup": 1.0}]
    offers = st.guard(state, ctx(), plan)
    assert [o["rfq_id"] for o in offers] == ["R3-02"]
