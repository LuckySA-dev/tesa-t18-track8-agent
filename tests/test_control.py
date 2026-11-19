"""Control page safety: .env writes, competition detection, practice logs in the hand-in folder, round ids."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import control  # noqa: E402
from main import round_order  # noqa: E402


def test_save_config_refuses_newlines(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MARKET_URL=http://a:8000\nUSE_LLM=0\n", encoding="utf-8")
    monkeypatch.setattr(control, "ENV_FILE", env)
    out = control.save_config({"MARKET_URL": "http://x:8000\nUSE_LLM=1"})
    assert out.get("error") and out["saved"] == []
    assert env.read_text(encoding="utf-8") == "MARKET_URL=http://a:8000\nUSE_LLM=0\n"


def test_save_config_writes_plain_value(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nSPEC_P=0.9\n", encoding="utf-8")
    monkeypatch.setattr(control, "ENV_FILE", env)
    monkeypatch.setenv("SPEC_P", "0.9")
    out = control.save_config({"SPEC_P": "0.5"})
    assert out["saved"] == ["SPEC_P"] and env.read_text(encoding="utf-8") == "# c\nSPEC_P=0.5\n"
    assert os.environ["SPEC_P"] == "0.5"


def test_is_competition():
    assert control.is_competition("http://h:8001")
    assert control.is_competition("http://h:9000", "competition")
    assert not control.is_competition("http://h:8000", "sandbox")
    assert not control.is_competition("http://h:8000")


def test_practice_logs_found(tmp_path):
    (tmp_path / "decisions.jsonl").write_text('{"event": "new_game", "market": "sandbox"}\n', encoding="utf-8")
    (tmp_path / "memory-sandbox-ab.json").write_text("{}", encoding="utf-8")
    (tmp_path / "clean.jsonl").write_text('{"event": "new_game", "market": "competition"}\n', encoding="utf-8")
    assert control.practice_in_hand_in(tmp_path) == ["decisions.jsonl", "memory-sandbox-ab.json"]


def test_round_order_never_raises():
    assert round_order(None) == -1
    assert round_order("P1") < round_order("R1") < round_order("R2") < round_order("R10")
    for rid in ("PR1", "ROUND-03", "R", "X"):
        round_order(rid)
