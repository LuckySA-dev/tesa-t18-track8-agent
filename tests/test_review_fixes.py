"""Code review 18 Nov 13:20: per-key LLM counter, end-of-game floor scope, rival-only crowding, LLM candidates
without banned pairs, competition priors."""
import random

import llm
import pattern
import pricing
import strategy as st


def test_rate_counter_is_per_key_and_hides_the_key():
    a, b = llm.calls_file("sk-T18-train"), llm.calls_file("sk-T18-test")
    assert a != b
    assert "sk-T18" not in a.name and "train" not in a.name


def test_own_offers_do_not_make_a_scarce_buyer_crowded():
    rnd = {"round_id": "R1", "market_summary": [{"rfq_id": "R1-01", "buyer": "1 S", "wanted_kg": 180,
                                                  "bought_kg": 120, "offers": 3}],
           "my_offers": [{"rfq_id": "R1-01"}, {"rfq_id": "R1-01"}]}
    assert pattern.buyer_kinds([rnd])["1 S"]["kind"] == "scarce"     # 1 rival offer for 3 bags


SPE = {"max_defect_pct": 3.0, "max_moisture_pct": 12.5, "min_screen": 16}
INS = {"max_defect_pct": 15.0, "max_moisture_pct": 15.0, "min_screen": None}


def _state(number):
    return {"round": {"kind": "real", "number": number}, "capacity_left_kg": 600,
            "lots": [{"lot_id": "L1", "kg_available": 600,
                      "sample": {"defect_pct": 1.0, "moisture_pct": 11.0, "screen": 17}}],
            "rfqs": [{"rfq_id": "R-01", "buyer": "1 S", "buyer_type": "SPECIALTY", "spec": SPE, "kg": 180, "bags": 3},
                     {"rfq_id": "R-02", "buyer": "6 I", "buyer_type": "INSTANT", "spec": INS, "kg": 600, "bags": 10}]}


def _ctx():
    return st.Context({"timeline": {"rounds": 8}}, [], {"L1": 120.0}, random.Random(0), None, None, 90)


def test_round7_low_floor_only_for_instant(monkeypatch):
    monkeypatch.setenv("PATTERN", "1")
    plan = [{"lot_id": "L1", "rfq_id": "R-01", "bags": 1, "markup": 0.82},
            {"lot_id": "L1", "rfq_id": "R-02", "bags": 1, "markup": 0.82}]
    k = {o["rfq_id"]: o["price_per_kg"] / 120.0 for o in st.guard(_state(7), _ctx(), plan)}
    assert k["R-01"] >= 0.85 - 1e-6                    # Specialty in round 7: normal floor
    assert k["R-02"] < 0.85                            # Instant in round 7: may go to 0.81
    k8 = {o["rfq_id"]: o["price_per_kg"] / 120.0 for o in st.guard(_state(8), _ctx(), plan)}
    assert k8["R-01"] < 0.85                           # last round: any buyer


def test_competition_prior_only_without_evidence(monkeypatch):
    monkeypatch.setitem(pricing.CLEAR_PRIORS, "4 B4", 0.99)
    lr = pricing.CeilingLearner()
    assert lr.competition("4 B4", "COMMERCIAL") == (0.99, pricing.CLEAR_PRIOR_WEIGHT)
    lr.observe({"round_id": "R1", "my_offers": [], "market_summary": [
        {"rfq_id": "R1-01", "buyer": "4 B4", "buyer_type": "COMMERCIAL", "wanted_kg": 600, "bought_kg": 300,
         "offers": 2}]}, {"L": 100.0})
    assert lr.competition("4 B4", "COMMERCIAL") == (None, 0.0)     # live evidence (under-filled) wins
