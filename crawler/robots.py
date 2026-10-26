"""
robots.txt parser that follows RFC 9309 / Google's matching rules.

Python's built-in urllib.robotparser does plain prefix matching and ignores the
`*` and `$` wildcards that Shopify and most large sites use (for example
`Disallow: /collections/*sort_by*`), so it silently over-permits. This module
implements the real rules:

* Pick the group whose user-agent token matches ours (case-insensitive,
  longest match wins); fall back to the `*` group.
* Within the group, the rule with the longest matching path wins; on a tie,
  Allow beats Disallow.
* `*` matches any run of characters, a trailing `$` anchors the end.
* A missing robots.txt (4xx) means everything is allowed; a server error (5xx)
  or network failure means we treat the whole site as disallowed for now.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit, unquote


@dataclass
class _Group:
    agents: list[str] = field(default_factory=list)
    rules: list[tuple[bool, str]] = field(default_factory=list)  # (allow, pattern)
    crawl_delay: float | None = None


def _compile(pattern: str) -> re.Pattern:
    anchored = pattern.endswith("$")
    if anchored:
        pattern = pattern[:-1]
    regex = "".join(".*" if ch == "*" else re.escape(ch) for ch in pattern)
    return re.compile(regex + ("$" if anchored else ""))


class RobotsPolicy:
    """Parsed robots.txt for one origin."""

    def __init__(self, text: str = "", *, allow_all: bool = False, disallow_all: bool = False):
        self.allow_all = allow_all
        self.disallow_all = disallow_all
        self.groups: list[_Group] = []
        self.sitemaps: list[str] = []
        self._compiled: dict[str, re.Pattern] = {}
        if text:
            self._parse(text)

    # ------------------------------------------------------------------ parse
    def _parse(self, text: str) -> None:
        current: _Group | None = None
        last_was_agent = False
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "user-agent":
                # Consecutive user-agent lines share one group.
                if current is None or not last_was_agent:
                    current = _Group()
                    self.groups.append(current)
                current.agents.append(value.lower())
                last_was_agent = True
                continue

            last_was_agent = False
            if key == "sitemap":
                self.sitemaps.append(value)
            elif current is None:
                continue
            elif key in ("allow", "disallow"):
                if value == "" and key == "disallow":
                    continue  # "Disallow:" with no path allows everything
                current.rules.append((key == "allow", value))
            elif key == "crawl-delay":
                try:
                    current.crawl_delay = float(value)
                except ValueError:
                    pass

    # ------------------------------------------------------------------ query
    def _group_for(self, user_agent: str) -> _Group | None:
        token = user_agent.split("/", 1)[0].strip().lower()
        best: _Group | None = None
        best_len = -1
        star: _Group | None = None
        for group in self.groups:
            for agent in group.agents:
                if agent == "*":
                    star = star or group
                elif token.startswith(agent) or agent.startswith(token):
                    if len(agent) > best_len:
                        best, best_len = group, len(agent)
        return best or star

    def _matches(self, pattern: str, path: str) -> bool:
        compiled = self._compiled.get(pattern)
        if compiled is None:
            compiled = self._compiled[pattern] = _compile(pattern)
        return compiled.match(path) is not None

    def can_fetch(self, user_agent: str, url: str) -> bool:
        if self.disallow_all:
            return False
        if self.allow_all:
            return True
        group = self._group_for(user_agent)
        if group is None:
            return True
        parts = urlsplit(url)
        path = unquote(parts.path or "/")
        if parts.query:
            path += "?" + parts.query

        verdict, verdict_len = True, -1
        for allow, pattern in group.rules:
            if self._matches(pattern, path):
                length = len(pattern)
                if length > verdict_len or (length == verdict_len and allow):
                    verdict, verdict_len = allow, length
        return verdict

    def crawl_delay(self, user_agent: str) -> float | None:
        group = self._group_for(user_agent)
        return group.crawl_delay if group else None
