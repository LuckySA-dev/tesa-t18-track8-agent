"""position.py: lot bands, pot, calendar weights, market balance, gaps, allocation."""
import position as pos

NOISE = {"defect": 0.35, "moisture": 0.30}
SPE = {"max_defect_pct": 3, "max_moisture_pct": 12.2, "min_screen": 16}
COM = {"max_defect_pct": 7, "max_moisture_pct": 12.5, "min_screen": None}
INS = {"max_defect_pct": 15, "max_moisture_pct": 14, "min_screen": None}
SPECS = {"SPE1": SPE, "COM3": COM, "INS5": INS}


def test_band_wide_near_grade_step_narrow_elsewhere():
    near = pos.lot_band({"defect_pct": 3.05, "moisture_pct": 11, "screen": 15}, None, NOISE, SPECS)
    far = pos.lot_band({"defect_pct": 2.0, "moisture_pct": 11, "screen": 15}, None, NOISE, SPECS)
    assert near["near_step"] and near["hi"] / near["lo"] > 1.035
    assert not far["near_step"] and far["err"] < 0.005
    assert near["lo"] <= near["q25"] <= near["mid"] <= near["hi"]
    assert pos.k_margin(near, "lo") < pos.k_margin(near, "q25") <= 1.0 == pos.k_margin(near, "mid")


def test_no_noise_band_is_a_point():
    b = pos.lot_band({"defect_pct": 1.79, "moisture_pct": 13.23, "screen": 14}, None, None, SPECS)
    assert b["lo"] == b["hi"] == b["mid"] == 119.75 and b["p_spec"]["COM3"] == 0 and b["p_spec"]["INS5"] == 1


def test_pot_and_pr_floor():
    lots = [{"lot_id": "A", "kg_total": 600, "kg_remaining": 300}, {"lot_id": "B", "kg_total": 300, "kg_remaining": 300}]
    bands = {"A": {"lo": 99, "mid": 100, "hi": 101}, "B": {"lo": 80, "mid": 80, "hi": 80}}
    p = pos.pot_band(lots, bands, revenue=30000)
    assert p["pot_mid"] == 84000 and p["bags_left"] == 10
    assert p["pr_so_far"] == 1.0                                  # 300 kg of A sold for 30,000 = k 1.00
    assert p["pr_floor"] == round((30000 + 0.8 * 54000) / 84000, 4)


def test_calendar_weights_live_format():
    cal = [{"rounds": "1-2", "label": "x", "demand_weight": 1.0, "specialty_share": None},
           {"rounds": "3–4", "label": "y", "demand_weight": 1.4, "specialty_share": 0.3}, {"rounds": "bad"}]
    w = pos.calendar_weights(cal, 5)
    assert w[1] == (1.0, None) and w[4] == (1.4, 0.3) and w[5] == (1.0, None)


def _round(rid, rows):
    return {"round_id": rid, "market_summary": [
        {"rfq_id": f"{rid}-{i}", "buyer": b, "buyer_type": t, "wanted_kg": w * 60, "bought_kg": got * 60, "offers": n}
        for i, (b, t, w, got, n) in enumerate(rows)]}


def test_market_balance_pressure_and_projection():
    rounds = [_round("R1", [("c", "COMMERCIAL", 100, 100, 190), ("i", "INSTANT", 90, 60, 150)])]
    weights = pos.calendar_weights([{"rounds": "3", "demand_weight": 2.0}], 4)
    b = pos.market_balance(rounds, n_teams=50, bags_per_team=20, round_number=2, rounds_total=4, weights=weights)
    assert b["demand_per_round"]["COMMERCIAL"] == 100 and b["demand_left"]["COMMERCIAL"] == 400   # 100 + 200 + 100
    assert b["supply_left"] == 1000 - 160 and b["pressure"] == round(840 / (400 + 360), 2)


def test_buyer_gaps_kinds():
    rounds = [_round("R1", [("ceil", "INSTANT", 20, 10, 36), ("scarce", "COMMERCIAL", 30, 30, 20),
                            ("crowd", "COMMERCIAL", 30, 30, 60), ("ok", "INSTANT", 20, 20, 22)])]
    g = pos.buyer_gaps(rounds, lambda r: r["buyer"])
    assert {k: v["kind"] for k, v in g.items()} == {"ceil": "ceiling", "scarce": "scarce", "crowd": "crowded", "ok": "balanced"}


def test_allocation_holds_best_lot_for_specialty():
    lots = [{"lot_id": "BEST", "kg_remaining": 300}, {"lot_id": "LOW", "kg_remaining": 360}]
    bands = {"BEST": pos.lot_band({"defect_pct": 1.0, "moisture_pct": 11.0, "screen": 17}, None, NOISE, SPECS),
             "LOW": pos.lot_band({"defect_pct": 10.0, "moisture_pct": 13.0, "screen": 13}, None, NOISE, SPECS)}
    by_type = {"SPECIALTY": ["SPE1"], "COMMERCIAL": ["COM3"], "INSTANT": ["INS5"]}
    pull = pos.allocation_pull(lots, bands, by_type, {"demand_left": {"SPECIALTY": 10, "COMMERCIAL": 20, "INSTANT": 20}})
    assert pull["BEST"]["hold_for_specialty"] and pull["BEST"]["reach"] == ["SPECIALTY", "COMMERCIAL", "INSTANT"]
    assert pull["LOW"]["reach"] == ["INSTANT"] and pull["LOW"]["single_outlet"]


def test_implied_k_back_solves_price():
    specs = {"COM": COM, "INS": INS}
    tmap = {"COM": "COMMERCIAL", "INS": "INSTANT"}
    mean, _ = pos.eligible_tv(COM, [], None)
    rnd = {"round_id": "R1", "market_summary": [
        {"rfq_id": "a", "buyer": "COM", "buyer_type": "COMMERCIAL", "wanted_kg": 600, "bought_kg": 600, "price_avg": mean * 0.98},
        {"rfq_id": "b", "buyer": "INS", "buyer_type": "INSTANT", "wanted_kg": 600, "bought_kg": 0, "price_avg": None}]}
    out = pos.implied_k([rnd], specs, lambda r: r["buyer"], tmap.get)
    assert out["COM"]["k"] == [0.98] and "INS" not in out          # no sale -> no k
