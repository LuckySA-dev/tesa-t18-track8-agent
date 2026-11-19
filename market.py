"""Client for every market REST endpoint (docs/คู่มือ-API.md): auth, retries, 429 back-off, readable errors."""
import time

import httpx


class MarketError(Exception):
    """An error the market returned: {"error": {code, message, field, hint, allowed}}."""

    def __init__(self, status: int, body: dict):
        err = body.get("error", {}) if isinstance(body, dict) else {}
        self.status = status
        self.code = err.get("code", f"HTTP_{status}")
        self.message = err.get("message", str(body))
        self.hint = err.get("hint")
        self.allowed = err.get("allowed")
        super().__init__(f"{self.code}: {self.message}" + (f" (hint: {self.hint})" if self.hint else ""))


def api_root(url: str) -> str:
    """The API lives at the server root (/v1/...). People copy the web page URL (http://host:8000/trade) into
    MARKET_URL; keep only scheme://host:port so both forms work (18 Nov: /trade/v1/state -> 404 everywhere)."""
    from urllib.parse import urlparse
    u = urlparse(url.strip())
    return f"{u.scheme}://{u.netloc}" if u.scheme and u.netloc else url.rstrip("/")


class Market:
    def __init__(self, base_url: str, key: str, timeout: float = 10.0):
        self.base_url = api_root(base_url)
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout,
                                 headers={"Authorization": f"Bearer {key}", "User-Agent": "coffee-starter/1.0"})

    def call(self, method: str, path: str, json=None, params=None, tries: int = 4, content=None, headers=None):
        for attempt in range(tries):
            try:
                r = self.http.request(method, path, json=json, params=params, content=content, headers=headers)
            except httpx.TransportError:            # server busy or network blip: wait and retry
                if attempt == tries - 1:
                    raise
                time.sleep(1 + attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == tries - 1:
                    break
                time.sleep(float(r.headers.get("Retry-After", 1 + attempt)))
                continue
            break
        body = r.json() if r.content else {}
        if r.status_code >= 400:
            raise MarketError(r.status_code, body)
        return body

    # ---- read
    def version(self) -> dict:                       # no key needed: server version, mode, server_time
        return self.call("GET", "/v1/version")

    def state(self) -> dict:                         # everything at once: phase, time, round, lots, RFQs, offers
        return self.call("GET", "/v1/state")

    def calendar(self) -> dict:                      # {"rules": TV formula + numbers, "calendar": round labels}
        return self.call("GET", "/v1/calendar")

    def lots(self):
        return self.call("GET", "/v1/lots")

    def rfqs(self):
        return self.call("GET", "/v1/rfqs")

    def appraisals(self) -> dict:                    # the appraisal we already sent
        return self.call("GET", "/v1/appraisals")

    def offers(self) -> dict:                        # our offers in the current round
        return self.call("GET", "/v1/offers")

    def results(self, round_id: str | None = None) -> dict:   # last cleared round, or ?round=R3
        return self.call("GET", "/v1/results", params={"round": round_id} if round_id else None)

    def standings(self) -> dict:                     # revenue, kg sold / left, average price
        return self.call("GET", "/v1/standings")

    def leaderboard(self) -> dict:                   # ranks only while the market runs
        return self.call("GET", "/v1/leaderboard")

    def profile(self) -> dict:
        return self.call("GET", "/v1/profile")

    # ---- write
    def appraise(self, items: list[dict]) -> dict:   # once, APPRAISAL phase only
        return self.call("POST", "/v1/appraisals", json={"items": items})

    def offer(self, o: dict) -> dict:                # {"rfq_id", "lot_id", "kg", "price_per_kg"} -> 201 OPEN
        return self.call("POST", "/v1/offers", json=o)

    def offer_batch(self, offers: list[dict], replace_existing: bool = True) -> dict:   # all or nothing
        return self.call("POST", "/v1/offers:batch", json={"offers": offers, "replace_existing": replace_existing})

    def update_offer(self, offer_id: str, kg: int | None = None, price_per_kg: float | None = None) -> dict:
        body = {k: v for k, v in (("kg", kg), ("price_per_kg", price_per_kg)) if v is not None}
        return self.call("PUT", f"/v1/offers/{offer_id}", json=body)

    def cancel_offer(self, offer_id: str) -> dict:
        return self.call("DELETE", f"/v1/offers/{offer_id}")

    def set_profile(self, team_name: str, university: str) -> dict:   # shown on the leaderboard and stage screen
        return self.call("POST", "/v1/profile", json={"team_name": team_name, "university": university})

    # ---- sandbox only
    def sandbox_reset(self) -> dict:                 # ends our sandbox game now, new lots
        return self.call("POST", "/v1/sandbox/reset")

    def sandbox_reveal(self) -> dict:                # after CLOSED: true quality, true TV, score
        return self.call("GET", "/v1/sandbox/reveal")

    # ---- code review
    def handin(self, zip_bytes: bytes) -> dict:      # only when staff call the team (pack.py --send does this)
        return self.call("POST", "/v1/handin", content=zip_bytes, headers={"Content-Type": "application/zip"}, tries=1)
