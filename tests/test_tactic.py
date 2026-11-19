"""TACTIC=1 controller: per-buyer feedback, modes, LLM clamp, spread, floor."""
import types

import tactic


def mem(rounds, sent=None):
    return types.SimpleNamespace(rounds=rounds, sent=sent or {}, tactic_mode=None)


def rnd(rid, offers, summary):
    return {"round_id": rid, "my_offers": offers, "market_summary": summary}


TV = {"L1": 100.0, "L2": 120.0}
P = tactic.params()


def test_above_limit_caps_buyer_and_steps_down():
    m = mem([rnd("R1", [{"rfq_id": "R1-01", "lot_id": "L1", "price_per_kg": 105.0, "kg": 60, "filled_kg": 0,
                        "status": "ABOVE_LIMIT"}],
                 [{"rfq_id": "R1-01", "buyer": "B3", "buyer_type": "COMMERCIAL", "price_max": None}])])
    nxt, by_type, cap, _, extra = tactic.feedback(m, TV, P)
    assert abs(extra["type_cap"]["COMMERCIAL"] - cap["B3"]) < 1e-9
    assert abs(cap["B3"] - (1.05 - P["cap_gap"])) < 1e-9
    assert abs(nxt["B3"] - (1.05 - P["down_above"])) < 1e-9
    assert by_type["COMMERCIAL"] == nxt["B3"]


def test_accepted_asks_more_outbid_follows_market():
    m = mem([rnd("R1", [{"rfq_id": "R1-01", "lot_id": "L1", "price_per_kg": 90.0, "kg": 60, "filled_kg": 60,
                         "status": "ACCEPTED"},
                        {"rfq_id": "R1-02", "lot_id": "L2", "price_per_kg": 120.0, "kg": 60, "filled_kg": 0,
                         "status": "OUTBID"}],
                 [{"rfq_id": "R1-01", "buyer": "B5", "buyer_type": "INSTANT"},
                  {"rfq_id": "R1-02", "buyer": "B3", "buyer_type": "COMMERCIAL", "price_max": 114.0}])])
    nxt, _, _, _, extra = tactic.feedback(m, TV, P)
    assert extra["type_acc"]["INSTANT"] == [(0, 0.9)]
    assert abs(nxt["B5"] - (0.90 + P["up"])) < 1e-9
    assert abs(nxt["B3"] - (114.0 / 120 - P["edge"])) < 1e-9          # market paid at most 0.95


def state(stock_bags, cap_bags):
    return {"lots": [{"lot_id": "L1", "kg_available": stock_bags * 60}], "round_capacity_kg": cap_bags * 60,
            "capacity_left_kg": cap_bags * 60,
            "rfqs": [{"rfq_id": "R5-01", "buyer": "B3", "buyer_type": "COMMERCIAL"},
                     {"rfq_id": "R5-02", "buyer": "B4", "buyer_type": "COMMERCIAL"}]}


def test_modes():
    hist_dry = [{"round": "R3", "real": True, "offered": 300, "sold": 0}, {"round": "R4", "real": True, "offered": 300, "sold": 0}]
    assert tactic.choose_mode(state(10, 5), hist_dry, 6, P, None)[0] == "RESCUE"
    assert tactic.choose_mode(state(10, 5), hist_dry[1:], 6, P, None)[0] == "SAFE"
    ok = [{"round": "R4", "real": True, "offered": 300, "sold": 300}]
    assert tactic.choose_mode(state(10, 5), ok, 6, P, None)[0] == "ATTACK"           # cover 0.33
    assert tactic.choose_mode(state(10, 5), ok, 6, P, "RESCUE")[0] == "SAFE"         # one step up per round
    assert tactic.choose_mode(state(3, 5), ok, 1, P, None)[0] == "SAFE"              # last round, stock fits
    assert tactic.choose_mode(state(10, 5), ok, 1, P, None)[0] == "RESCUE"           # last round, stock > capacity


def test_sell_limit_risk():
    ok = [{"round": "R1", "real": True, "offered": 300, "sold": 300}]
    r = tactic.risk(stock=20, cap=5, rounds_left=4, hist=ok, prior=0.7)   # rate 0.85 -> 4.25 bags / round
    assert r["rate"] == 0.85 and abs(r["need"] - 4.7) < 0.05
    assert r["unsold"] > 0.1                       # 20 bags cannot go through 4 rounds of 4.25
    assert r["if_miss"] > r["unsold"]              # missing a round makes it worse: the limit does not carry over
    r = tactic.risk(stock=8, cap=5, rounds_left=6, hist=ok, prior=0.7)
    assert r["unsold"] == 0 and r["if_miss"] == 0 and r["spare"] > 3


def test_llm_clamp_cap_and_floor():
    ctx = types.SimpleNamespace(memory=mem([]), tv=TV, notes=[])
    planner = [{"lot_id": "L1", "rfq_id": "R5-01", "bags": 2, "markup": 0.97}]
    llm = [{"lot_id": "L1", "rfq_id": "R5-01", "bags": 2, "markup": 1.05}]
    out = tactic.apply(state(10, 5), ctx, llm, planner, 6, False)
    assert out[0]["markup"] <= 0.97 + P["llm_delta"] + P["shift"]["ATTACK"] + 1e-9
    low = [{"lot_id": "L1", "rfq_id": "R5-01", "bags": 2, "markup": 0.5}]
    out = tactic.apply(state(10, 5), ctx, low, low, 6, False)
    assert out[0]["markup"] >= tactic.SALVAGE + P["floor_gap"] - 1e-9


def test_spread_keeps_one_rfq_from_taking_the_round():
    rows = [{"lot_id": "L1", "rfq_id": "R5-01", "bags": 5, "markup": 1.0},
            {"lot_id": "L1", "rfq_id": "R5-02", "bags": 5, "markup": 1.0}]
    out = tactic.spread(rows, state(10, 5), P)
    assert max(r["bags"] for r in out if r["rfq_id"] == "R5-01") == 3                 # ceil(5 x 0.6)


def test_anchor_follows_what_buyers_paid():
    m = mem([rnd("R1", [], [{"rfq_id": "R1-01", "buyer": "B3", "buyer_type": "COMMERCIAL", "wanted_kg": 600,
                             "bought_kg": 600, "price_avg": 98.0, "price_max": 104.0},
                            {"rfq_id": "R1-02", "buyer": "B5", "buyer_type": "INSTANT", "wanted_kg": 600,
                             "bought_kg": 300, "price_avg": 85.0, "price_max": 92.0}])])
    m.buyers = {"B3": {"spec": {"max_defect_pct": 7, "max_moisture_pct": 12.5}},
                "B5": {"spec": {"max_defect_pct": 15, "max_moisture_pct": 14}}}
    lots = [{"lot_id": "L1", "sample": {"defect_pct": 3, "moisture_pct": 11, "screen": 15}}]
    a = tactic.anchors(m, lots, TV, P)
    assert a["B3"]["k"] == 0.98 and a["B3"]["filled"]          # filled RFQ: average price sets the level
    assert a["B5"]["k"] == 0.92 and not a["B5"]["filled"]      # under-filled: its top price (ceiling binds)


def test_lot_tv_correction_only_when_other_lots_sold():
    def r(idx, offers):
        return rnd(f"R{idx}", offers, [{"rfq_id": "X", "buyer": "B6", "buyer_type": "INSTANT"}])
    above = {"rfq_id": "X", "lot_id": "L1", "price_per_kg": 85.0, "kg": 60, "filled_kg": 0, "status": "ABOVE_LIMIT"}
    sold = {"rfq_id": "X", "lot_id": "L2", "price_per_kg": 102.0, "kg": 60, "filled_kg": 60, "status": "ACCEPTED"}
    _, _, _, _, ex = tactic.feedback(mem([r(1, [above, sold])]), TV, P)          # L2 sold at 0.85 too (factor is
    # computed always; apply() uses it only with TAC_LOT_TV=1)
    assert abs(ex["lot_factor"]["L1"] - (1 - P["lot_cut"])) < 1e-9
    _, _, _, _, ex = tactic.feedback(mem([r(1, [above])]), TV, P)                # nobody else sold: ceiling binds
    assert "L1" not in ex["lot_factor"]
