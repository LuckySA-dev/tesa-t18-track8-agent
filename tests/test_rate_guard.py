"""llm.RateGuard: organiser gateway limits (<= 20 calls / min per team, no bursts, cooldown after 429)."""
from llm import LLM, RateGuard


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def guard(tmp_path, clock, per_min=10, gap=4.0):
    return RateGuard(tmp_path / "calls.json", max_per_min=per_min, min_gap=gap, clock=clock, sleep=clock.sleep)


def test_min_gap_waits_briefly_or_refuses(tmp_path):
    c = Clock()
    g = guard(tmp_path, c)
    assert g.acquire(max_wait=5) is None
    assert g.acquire(max_wait=5) is None            # waited the 4 s gap
    assert c.t == 1004.0
    assert g.acquire(max_wait=1) is not None        # would need 4 s more: refused, not recorded
    c.t += 4
    assert g.acquire(max_wait=0) is None


def test_per_minute_cap(tmp_path):
    c = Clock()
    g = guard(tmp_path, c, per_min=10, gap=0)
    for _ in range(10):
        assert g.acquire(max_wait=0) is None
        c.t += 1
    assert g.acquire(max_wait=5) is not None        # 11th inside the same minute
    c.t = 1060.5                                    # first call left the window
    assert g.acquire(max_wait=0) is None


def test_shared_between_instances(tmp_path):
    c = Clock()
    a, b = guard(tmp_path, c), guard(tmp_path, c)   # e.g. main.py and check.py on the same PC
    assert a.acquire(max_wait=0) is None
    assert b.acquire(max_wait=0) is not None


def test_cooldown_after_429(tmp_path):
    c = Clock()
    g = guard(tmp_path, c, gap=0)
    g.cooldown(60)
    assert g.acquire(max_wait=5) is not None
    c.t += 61
    assert g.acquire(max_wait=0) is None


class FakeCompletions:
    def __init__(self, error):
        self.calls, self.error = 0, error

    def create(self, **kw):
        self.calls += 1
        raise self.error


class Err(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.status_code = code


def make_llm(tmp_path, error):
    llm = LLM("http://127.0.0.1:9/v1", "event-llm", "x")
    c = Clock()
    llm.guard = guard(tmp_path, c, gap=0)
    comp = FakeCompletions(error)
    fake = type("C", (), {"chat": type("Ch", (), {"completions": comp})()})()
    llm.client = type("W", (), {"with_options": lambda self, **kw: fake})()
    return llm, comp


def test_at_most_two_sends_per_request(tmp_path):
    llm, comp = make_llm(tmp_path, Err(400))        # every thinking style rejected
    assert llm.ask_json("s", "u", {}) is None
    assert comp.calls <= 2


def test_429_does_not_retry_and_blocks(tmp_path):
    llm, comp = make_llm(tmp_path, Err(429))
    assert llm.ask_json("s", "u", {}) is None
    assert comp.calls == 1
    assert llm.ask_json("s", "u", {}) is None       # cooldown: not even sent
    assert comp.calls == 1
    assert "rate guard" in llm.last_error
