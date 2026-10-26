"""
Polite HTTP client used by every adapter.

* Checks robots.txt before every request (and refuses if disallowed).
* Enforces a minimum delay between requests to the same host, raised to the
  site's Crawl-delay if it asks for more.
* Identifies itself with a descriptive User-Agent that includes a contact.
* Backs off on 429/503 (honouring Retry-After) and gives up on a site that keeps
  refusing — it never tries to get around blocking.
* Optional on-disk cache so re-running during development doesn't hit sites
  again for pages fetched in the last N hours.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import requests

from .robots import RobotsPolicy


class SiteBlocked(Exception):
    """The site refused us (403, repeated 429, or robots.txt disallows). Stop crawling it."""


class RobotsDisallowed(Exception):
    """robots.txt disallows this specific URL."""


@dataclass
class FetchResult:
    url: str
    status: int
    text: str
    from_cache: bool = False

    def json(self):
        return json.loads(self.text)


class PoliteSession:
    def __init__(
        self,
        user_agent: str,
        min_delay: float = 2.0,
        timeout: float = 20.0,
        cache_dir: str | None = None,
        cache_max_age_hours: float = 12.0,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.user_agent = user_agent
        self.min_delay = min_delay
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.cache_max_age = cache_max_age_hours * 3600
        self.max_retries = max_retries
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
        })
        self._robots: dict[str, RobotsPolicy] = {}
        self._last_request: dict[str, float] = {}
        self._host_delay: dict[str, float] = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------- helpers
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _origin(url: str) -> str:
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.netloc}"

    def set_host_delay(self, url: str, seconds: float) -> None:
        """Override the per-host delay (e.g. a CDN can take a shorter delay)."""
        self._host_delay[urlsplit(url).netloc] = seconds

    def _wait_turn(self, host: str) -> None:
        delay = self._host_delay.get(host, self.min_delay)
        elapsed = time.monotonic() - self._last_request.get(host, 0.0)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request[host] = time.monotonic()

    # -------------------------------------------------------------- robots
    def robots_for(self, url: str) -> RobotsPolicy:
        origin = self._origin(url)
        if origin in self._robots:
            return self._robots[origin]
        robots_url = origin + "/robots.txt"
        host = urlsplit(url).netloc
        self._wait_turn(host)
        try:
            resp = self.session.get(robots_url, timeout=self.timeout)
            if resp.status_code >= 500:
                policy = RobotsPolicy(disallow_all=True)
                self.log(f"  robots.txt at {origin} returned {resp.status_code}; treating site as off-limits for now")
            elif resp.status_code >= 400:
                policy = RobotsPolicy(allow_all=True)
            else:
                policy = RobotsPolicy(resp.text)
        except requests.RequestException as exc:
            self.log(f"  could not read {robots_url} ({exc}); treating site as off-limits for now")
            policy = RobotsPolicy(disallow_all=True)

        site_delay = policy.crawl_delay(self.user_agent)
        if site_delay and site_delay > self._host_delay.get(host, self.min_delay):
            self._host_delay[host] = site_delay
            self.log(f"  {host} asks for Crawl-delay {site_delay}s — honouring it")
        self._robots[origin] = policy
        return policy

    def allowed(self, url: str) -> bool:
        return self.robots_for(url).can_fetch(self.user_agent, url)

    # --------------------------------------------------------------- cache
    def _cache_path(self, url: str) -> str | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return os.path.join(self.cache_dir, digest + ".json")

    def _cache_read(self, url: str) -> dict | None:
        path = self._cache_path(url)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_write(self, url: str, resp: requests.Response) -> None:
        path = self._cache_path(url)
        if not path:
            return
        entry = {
            "url": url,
            "fetched_at": time.time(),
            "status": resp.status_code,
            "etag": resp.headers.get("ETag"),
            "last_modified": resp.headers.get("Last-Modified"),
            "text": resp.text,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entry, fh)
        os.replace(tmp, path)

    # ---------------------------------------------------------------- fetch
    def get(self, url: str, *, use_cache: bool = True) -> FetchResult:
        if not self.allowed(url):
            raise RobotsDisallowed(url)

        cached = self._cache_read(url) if use_cache else None
        if cached and time.time() - cached["fetched_at"] < self.cache_max_age:
            return FetchResult(url, cached["status"], cached["text"], from_cache=True)

        headers = {}
        if cached:
            if cached.get("etag"):
                headers["If-None-Match"] = cached["etag"]
            if cached.get("last_modified"):
                headers["If-Modified-Since"] = cached["last_modified"]

        host = urlsplit(url).netloc
        attempt = 0
        while True:
            self._wait_turn(host)
            try:
                resp = self.session.get(url, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                self.log(f"  network error on {url} ({exc}); retry {attempt}/{self.max_retries}")
                time.sleep(min(60, 2 ** attempt))
                continue

            if resp.status_code == 304 and cached:
                cached["fetched_at"] = time.time()
                path = self._cache_path(url)
                if path:
                    with open(path, "w", encoding="utf-8") as fh:
                        json.dump(cached, fh)
                return FetchResult(url, 200, cached["text"], from_cache=True)

            if resp.status_code in (429, 503):
                attempt += 1
                if attempt > self.max_retries:
                    raise SiteBlocked(f"{host} keeps answering {resp.status_code}; backing off from this site")
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else min(120, 10 * 2 ** attempt)
                # Slow down for the rest of the crawl too.
                self._host_delay[host] = max(self._host_delay.get(host, self.min_delay) * 2, wait / 2)
                self.log(f"  {host} said {resp.status_code}; waiting {wait:.0f}s and slowing down")
                time.sleep(wait)
                continue

            if resp.status_code in (401, 403):
                raise SiteBlocked(f"{host} refused access ({resp.status_code}); skipping this site")

            if resp.ok:
                self._cache_write(url, resp)
            return FetchResult(url, resp.status_code, resp.text)

    def get_bytes(self, url: str) -> bytes | None:
        """Fetch binary content (images). Not cached here; callers store the file."""
        if not self.allowed(url):
            raise RobotsDisallowed(url)
        host = urlsplit(url).netloc
        for attempt in range(1, self.max_retries + 1):
            self._wait_turn(host)
            try:
                resp = self.session.get(url, timeout=self.timeout, headers={"Accept": "image/*"})
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (429, 503):
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            if resp.status_code in (401, 403):
                raise SiteBlocked(f"{host} refused image access ({resp.status_code})")
            return resp.content if resp.ok else None
        return None
