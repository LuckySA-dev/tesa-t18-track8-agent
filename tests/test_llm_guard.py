"""main.llm_allowed: in a scored market only the event gateway may be called (anything else = disqualification)."""
import pytest

from main import llm_allowed

GW, OLLAMA = "http://202.29.227.22:4000/v1", "http://localhost:11434/v1"
TRAIN, TEST = "http://202.29.227.22:8000", "http://202.29.227.22:8001"


@pytest.mark.parametrize("url, mode, base, model, ok", [
    (TRAIN, "sandbox", OLLAMA, "qwen3.5:9b", True),          # practice: Ollama is fine
    (TRAIN, "sandbox", GW, "event-llm", True),
    ("http://127.0.0.1:8899", "sandbox", OLLAMA, "qwen3.5:9b", True),   # local mock market
    (TEST, "competition", OLLAMA, "qwen3.5:9b", False),      # the mistake this guard exists for
    (TEST, "sandbox", OLLAMA, "qwen3.5:9b", False),          # port 8001 is scored whatever mode says
    (TEST, None, OLLAMA, "qwen3.5:9b", False),               # /v1/version failed: assume scored
    (TRAIN, "competition", OLLAMA, "qwen3.5:9b", False),     # train port switched to a scored stage
    (TEST, "competition", GW, "event-llm", True),            # the allowed setup
    (TEST, "competition", GW, "event-11m", False),           # endpoint.txt typo
    (TEST, "competition", "http://202.29.227.22:4001/v1", "event-llm", False),
    (TEST, "competition", "", "", False),
])
def test_llm_allowed(url, mode, base, model, ok):
    assert llm_allowed(url, mode, base, model)[0] is ok
