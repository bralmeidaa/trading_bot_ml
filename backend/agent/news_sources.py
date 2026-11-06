"""
Free, no-signup text sources for the AI agent — this is the whole point of
using an LLM here: classical ML over OHLCV already proved to have no edge in
this project's research, but it never had access to qualitative/textual
context. Two sources for v1, both free and keyless:

1. Binance announcements — official JSON endpoint, no auth. Directly relevant
   (listing/delisting/maintenance/incident news for the exact instruments
   traded).
2. CoinTelegraph RSS — broad market narrative/headlines.

Deferred (not v1): CryptoPanic (needs a signup token), Twitter/X (paid tier),
on-chain narrative sources — see the plan.

Network calls here are meant to be wrapped in asyncio.to_thread by callers.
Every fetcher degrades to an empty list on any failure — a news outage must
never crash the agent's cycle, it should just mean less context that cycle.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

BINANCE_ANNOUNCEMENTS_URL = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
)
COINTELEGRAPH_RSS_URL = "https://cointelegraph.com/rss"

_REQUEST_TIMEOUT = 10


def fetch_binance_announcements(hours: int = 24) -> List[dict]:
    """Official Binance announcements from the last `hours`. [{title, published_at, url, source}]."""
    cutoff_ms = int(time.time() * 1000) - hours * 3600 * 1000
    try:
        resp = requests.get(
            BINANCE_ANNOUNCEMENTS_URL,
            params={"type": 1, "pageNo": 1, "pageSize": 20},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        articles = (data.get("data") or {}).get("catalogs", [])
        items = []
        for catalog in articles:
            for art in catalog.get("articles", []):
                ts = art.get("releaseDate", 0)
                if ts < cutoff_ms:
                    continue
                code = art.get("code", "")
                items.append({
                    "title": art.get("title", ""),
                    "published_at": ts,
                    "url": f"https://www.binance.com/en/support/announcement/{code}",
                    "source": "binance_announcements",
                })
        return items
    except Exception as exc:
        logger.warning(f"fetch_binance_announcements failed: {exc}")
        return []


def fetch_cointelegraph_rss(hours: int = 24) -> List[dict]:
    """CoinTelegraph RSS headlines from the last `hours`."""
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser not installed — skipping CoinTelegraph RSS")
        return []
    try:
        feed = feedparser.parse(COINTELEGRAPH_RSS_URL)
        cutoff = time.time() - hours * 3600
        items = []
        for entry in feed.entries:
            published = entry.get("published_parsed")
            ts = time.mktime(published) if published else time.time()
            if ts < cutoff:
                continue
            items.append({
                "title": entry.get("title", ""),
                "published_at": int(ts * 1000),
                "url": entry.get("link", ""),
                "source": "cointelegraph_rss",
            })
        return items
    except Exception as exc:
        logger.warning(f"fetch_cointelegraph_rss failed: {exc}")
        return []


def _base_asset(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC', also matching common long-forms."""
    return symbol.split("/")[0].upper()


_LONG_FORMS = {"BTC": ["bitcoin"], "ETH": ["ethereum", "ether"], "SOL": ["solana"]}


def digest_for_symbols(symbols: List[str], hours: int = 24,
                       max_items: int = 15) -> List[dict]:
    """Merge both sources, dedupe by URL, keep items mentioning one of the
    traded symbols OR broad-market items (untagged — still useful context),
    sorted newest-first, capped to bound prompt size."""
    items = fetch_binance_announcements(hours) + fetch_cointelegraph_rss(hours)

    seen_urls = set()
    deduped = []
    for item in items:
        url = item.get("url")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(item)

    keywords = set()
    for sym in symbols:
        base = _base_asset(sym)
        keywords.add(base.lower())
        keywords.update(_LONG_FORMS.get(base, []))

    def _tag(item: dict) -> dict:
        title_lower = item["title"].lower()
        item = dict(item)
        item["symbols_mentioned"] = [s for s in symbols
                                     if _base_asset(s).lower() in title_lower
                                     or any(lf in title_lower for lf in _LONG_FORMS.get(_base_asset(s), []))]
        return item

    deduped = [_tag(item) for item in deduped]
    deduped.sort(key=lambda x: x.get("published_at", 0), reverse=True)
    return deduped[:max_items]
