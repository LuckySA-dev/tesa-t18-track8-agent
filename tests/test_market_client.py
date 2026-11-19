"""market.Market calls the method + path in docs/คู่มือ-API.md for every endpoint (offline, mocked HTTP)."""
import httpx
import pytest

from market import Market, MarketError


def client(calls, status=200, body=None):
    def handler(req: httpx.Request):
        calls.append((req.method, req.url.path, req.url.query.decode(), req.content))
        return httpx.Response(status, json=body if body is not None else {})
    api = Market("http://m", "sk-T18-x")
    api.http = httpx.Client(base_url="http://m", transport=httpx.MockTransport(handler))
    return api


@pytest.mark.parametrize("fn, method, path", [
    (lambda a: a.version(), "GET", "/v1/version"),
    (lambda a: a.state(), "GET", "/v1/state"),
    (lambda a: a.calendar(), "GET", "/v1/calendar"),
    (lambda a: a.lots(), "GET", "/v1/lots"),
    (lambda a: a.rfqs(), "GET", "/v1/rfqs"),
    (lambda a: a.appraisals(), "GET", "/v1/appraisals"),
    (lambda a: a.appraise([{"lot_id": "L", "est_value_per_kg": 100}]), "POST", "/v1/appraisals"),
    (lambda a: a.offer({"rfq_id": "R1-01"}), "POST", "/v1/offers"),
    (lambda a: a.offer_batch([]), "POST", "/v1/offers:batch"),
    (lambda a: a.offers(), "GET", "/v1/offers"),
    (lambda a: a.update_offer("O1", price_per_kg=118.0), "PUT", "/v1/offers/O1"),
    (lambda a: a.cancel_offer("O1"), "DELETE", "/v1/offers/O1"),
    (lambda a: a.results(), "GET", "/v1/results"),
    (lambda a: a.standings(), "GET", "/v1/standings"),
    (lambda a: a.leaderboard(), "GET", "/v1/leaderboard"),
    (lambda a: a.profile(), "GET", "/v1/profile"),
    (lambda a: a.set_profile("T", "U"), "POST", "/v1/profile"),
    (lambda a: a.sandbox_reset(), "POST", "/v1/sandbox/reset"),
    (lambda a: a.sandbox_reveal(), "GET", "/v1/sandbox/reveal"),
    (lambda a: a.handin(b"PK"), "POST", "/v1/handin"),
])
def test_every_endpoint(fn, method, path):
    calls = []
    fn(client(calls))
    assert calls[0][:2] == (method, path)


def test_bodies_and_query():
    calls = []
    api = client(calls)
    api.results("R3")
    api.update_offer("O1", price_per_kg=118.0)
    api.offer_batch([{"rfq_id": "R1-01"}])
    assert calls[0][2] == "round=R3"
    assert calls[1][3] == b'{"price_per_kg":118.0}'                      # only the field that changes
    assert b'"replace_existing":true' in calls[2][3]


def test_error_is_readable():
    api = client([], 400, {"error": {"code": "NOT_MULTIPLE_OF_BAG", "message": "kg must be a multiple of 60",
                                     "hint": "use 60", "allowed": {"nearest": 60}}})
    with pytest.raises(MarketError) as e:
        api.offer({"kg": 61})
    assert e.value.code == "NOT_MULTIPLE_OF_BAG" and e.value.allowed == {"nearest": 60}


def test_market_url_with_trade_page_path_still_hits_api_root():
    from market import api_root
    assert api_root("http://202.29.227.22:8000/trade") == "http://202.29.227.22:8000"
    assert api_root("http://202.29.227.22:8001/trade/") == "http://202.29.227.22:8001"
    assert api_root("http://202.29.227.22:8000") == "http://202.29.227.22:8000"
    assert Market("http://m:8000/trade", "k").base_url == "http://m:8000"
