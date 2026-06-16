from __future__ import annotations

import math
import re
import time
from collections import Counter
import urllib.parse
from datetime import datetime, timezone
from urllib.parse import quote_plus

import pandas as pd
import requests

from .config import RiskEvent, SupplierProfile, now_iso

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
GDELT_ENDPOINT    = "https://api.gdeltproject.org/api/v2/doc/doc"
USGS_SIGNIFICANT  = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_hour.geojson"
USGS_4_5_DAY      = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
DDG_SEARCH        = "https://html.duckduckgo.com/html/"
GNEWS_RSS         = "https://gnews.io/api/v4/search"          # free tier, no key needed for RSS fallback
BING_NEWS_RSS     = "https://www.bing.com/news/search"

# How many seconds to wait between outbound requests to avoid rate-limiting
_REQUEST_DELAY = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_ts(value: str | None) -> str:
    """Convert heterogeneous timestamps into a single UTC ISO format."""
    if not value:
        return now_iso()
    try:
        v = str(value)
        if v.endswith("Z"):
            v = v.replace("Z", "+00:00")
        return datetime.fromisoformat(v).astimezone(timezone.utc).isoformat()
    except Exception:
        return now_iso()


def _clean_text(text: str, limit: int = 160) -> str:
    """Strip noise and trim the text for compact evidence cards."""
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def _score_from_text(text: str, keywords: list[str], base: float = 40.0, cap: float = 95.0) -> float:
    """Heuristic severity before the risk engine fuses everything."""
    lower = (text or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    if hits == 0:
        return base * 0.35
    return min(cap, base + 12 * hits)


def _get(url: str, params: dict | None = None, timeout: int = 20) -> requests.Response | None:
    """Resilient GET wrapper — returns None on any error so callers never crash."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source 1 — GDELT  (fixed query format + mode)
# ---------------------------------------------------------------------------


from collections import Counter

def _event_ts(value) -> str:
    if value is None:
        return ""
    try:
        if hasattr(value, "published_at"):
            value = getattr(value, "published_at")
        return str(value)
    except Exception:
        return ""

def _event_recent_key(ev) -> tuple:
    # Newest first, then strongest signal, then stable text fallback.
    ts = _event_ts(getattr(ev, "published_at", None))
    sev = float(getattr(ev, "severity", 0.0) or 0.0)
    title = str(getattr(ev, "title", "") or "")
    return (ts, sev, title)

def _compact_events(records: list, max_items: int = 15, per_category: int = 4):
    """
    Keep only the newest/highest-signal items and cap each category.
    Accepts either RiskEvent objects or dicts with the same keys.
    """
    buckets: dict[str, int] = {}
    picked = []
    for ev in sorted(records, key=_event_recent_key, reverse=True):
        cat = getattr(ev, "category", None) or (ev.get("category") if isinstance(ev, dict) else "unknown")
        if buckets.get(cat, 0) >= per_category:
            continue
        buckets[cat] = buckets.get(cat, 0) + 1
        picked.append(ev)
        if len(picked) >= max_items:
            break
    return picked

def summarize_events_for_llm(events: list, max_items: int = 8) -> dict:
    """
    Very small digest to keep prompts under the model context limit.
    """
    compact = _compact_events(events, max_items=max_items)
    counts = Counter(
        (getattr(ev, "category", None) if not isinstance(ev, dict) else ev.get("category")) or "unknown"
        for ev in compact
    )
    def _get(ev, key, default=None):
        if isinstance(ev, dict):
            return ev.get(key, default)
        return getattr(ev, key, default)

    return {
        "total_events_seen": len(events),
        "events_kept": len(compact),
        "by_category": dict(counts),
        "most_recent": [
            {
                "category": _get(ev, "category"),
                "title": _get(ev, "title"),
                "published_at": _get(ev, "published_at"),
                "severity": round(float(_get(ev, "severity", 0.0) or 0.0), 1),
                "source": _get(ev, "source"),
            }
            for ev in compact
        ],
    }


def search_gdelt(query: str, max_records: int = 6) -> list[dict]:
    """
    GDELT v2 Article List.  Key fixes vs original:
      - Use  mode=ArtList  +  sort=DateDesc  (HybridRel silently drops results for narrow queries)
      - Strip parentheses from the query; GDELT v2 uses plain AND/OR only
      - Fall through to empty list gracefully on any HTTP / JSON error
    """
    # GDELT rejects parenthesised grouping — convert to plain AND string
    clean_q = re.sub(r"[()\"]", " ", query).strip()
    clean_q = re.sub(r"\s+", " ", clean_q)

    params = {
        "query":      clean_q,
        "mode":       "ArtList",
        "format":     "json",
        "maxrecords": max_records,
        "sort":       "DateDesc",       # ← was HybridRel, which often yields 0 results
        "timespan":   "1month",         # ← keep it recent; no timespan = very old articles surface
    }

    resp = _get(GDELT_ENDPOINT, params=params, timeout=25)
    if resp is None:
        return []

    try:
        payload = resp.json()
    except Exception:
        return []

    articles = payload.get("articles") or []
    rows = []
    for art in articles:
        rows.append({
            "source":       "GDELT",
            "title":        art.get("title") or art.get("seendate") or "Untitled",
            "url":          art.get("url") or "",
            "published_at": _safe_ts(art.get("seendate")),
            "snippet":      _clean_text(art.get("snippet") or art.get("title") or ""),
            "domain":       art.get("domain") or "",
            "language":     art.get("language") or "",
            "sourcecountry":art.get("sourcecountry") or "",
        })
    return rows


# ---------------------------------------------------------------------------
# Source 2 — DuckDuckGo News  (no API key, scrapes HTML news results)
# ---------------------------------------------------------------------------

def search_ddg_news(query: str, max_records: int = 5) -> list[dict]:
    """
    DuckDuckGo HTML news scraper — zero API key required.
    Works inside the AMD environment (outbound HTTP is allowed).
    Uses the news vertical (kl=us-en, ia=news).
    """
    try:
        from html.parser import HTMLParser

        class _DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results: list[dict] = []
                self._cur: dict | None = None
                self._in_title = False
                self._in_snippet = False

            def handle_starttag(self, tag, attrs):
                a = dict(attrs)
                if tag == "div" and "result__body" in a.get("class", ""):
                    self._cur = {"title": "", "url": "", "snippet": "", "published_at": now_iso()}
                if tag == "a" and "result__a" in a.get("class", "") and self._cur is not None:
                    href = a.get("href", "")
                    # DDG wraps with redirect; extract uddg param
                    if "uddg=" in href:
                        raw = urllib.parse.urlparse(href)
                        qs  = urllib.parse.parse_qs(raw.query)
                        href = urllib.parse.unquote(qs.get("uddg", [""])[0])
                    self._cur["url"] = href
                    self._in_title = True
                if tag == "a" and "result__snippet" in a.get("class", "") and self._cur is not None:
                    self._in_snippet = True
                if tag == "span" and "result__check" in a.get("class", "") and self._cur is not None:
                    pass  # timestamp placeholder

            def handle_data(self, data):
                if self._in_title and self._cur is not None:
                    self._cur["title"] += data
                if self._in_snippet and self._cur is not None:
                    self._cur["snippet"] += data

            def handle_endtag(self, tag):
                if tag == "a":
                    self._in_title   = False
                    self._in_snippet = False
                if tag == "div" and self._cur and self._cur.get("title"):
                    self.results.append(self._cur)
                    self._cur = None

        resp = requests.post(
            DDG_SEARCH,
            data={"q": query + " site:reuters.com OR site:bloomberg.com OR site:ft.com OR site:bbc.com OR site:wsj.com OR site:apnews.com",
                  "kl": "us-en", "ia": "news"},
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
            timeout=20,
            allow_redirects=True,
        )
        resp.raise_for_status()

        parser = _DDGParser()
        parser.feed(resp.text)

        rows = []
        for r in parser.results[:max_records]:
            title   = _clean_text(r.get("title") or "")
            snippet = _clean_text(r.get("snippet") or "")
            if not title:
                continue
            rows.append({
                "source":       "DuckDuckGo",
                "title":        title,
                "url":          r.get("url") or "",
                "published_at": r.get("published_at") or now_iso(),
                "snippet":      snippet,
                "domain":       "",
                "language":     "en",
                "sourcecountry":"",
            })
        return rows

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source 3 — Bing News RSS  (no API key, public RSS feed)
# ---------------------------------------------------------------------------

def search_bing_news_rss(query: str, max_records: int = 5) -> list[dict]:
    """
    Bing News RSS — completely free, no key, machine-readable XML.
    Falls back gracefully to [] on any parse error.
    """
    try:
        import xml.etree.ElementTree as ET

        params = {"q": query, "format": "rss", "count": max_records}
        resp = _get(BING_NEWS_RSS, params=params, timeout=20)
        if resp is None:
            return []

        root = ET.fromstring(resp.text)
        ns   = {"media": "http://search.yahoo.com/mrss/"}
        rows = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            url   = (item.findtext("link")  or "").strip()
            desc  = (item.findtext("description") or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            if not title:
                continue
            # Convert RFC-2822 → ISO
            try:
                from email.utils import parsedate_to_datetime
                ts = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat()
            except Exception:
                ts = now_iso()
            rows.append({
                "source":       "BingNews",
                "title":        _clean_text(title),
                "url":          url,
                "published_at": ts,
                "snippet":      _clean_text(re.sub("<[^>]+>", "", desc)),
                "domain":       "",
                "language":     "en",
                "sourcecountry":"",
            })
        return rows[:max_records]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source 4 — USGS Earthquakes  (structured + reliable)
# ---------------------------------------------------------------------------

def get_usgs_significant_quakes() -> list[dict]:
    """
    Pull from both the significant-hour feed AND the 4.5+ day feed.
    The hour feed is often empty; combining both ensures logistics data.
    """
    rows = []
    for url in [USGS_SIGNIFICANT, USGS_4_5_DAY]:
        resp = _get(url, timeout=20)
        if resp is None:
            continue
        try:
            payload = resp.json()
        except Exception:
            continue
        for feat in payload.get("features", []) or []:
            props = feat.get("properties", {})
            geom  = feat.get("geometry", {})
            coords = geom.get("coordinates", [None, None, None])
            epoch  = props.get("time")
            ts = _safe_ts(
                datetime.fromtimestamp(epoch / 1000, tz=timezone.utc).isoformat() if epoch else None
            )
            rows.append({
                "source":       "USGS",
                "title":        props.get("title") or "Earthquake",
                "url":          props.get("url") or "",
                "published_at": ts,
                "snippet":      _clean_text(props.get("place") or props.get("title") or ""),
                "magnitude":    props.get("mag"),
                "place":        props.get("place"),
                "latitude":     coords[1],
                "longitude":    coords[0],
                "depth_km":     coords[2],
                "alert":        props.get("alert"),
            })

    # Deduplicate by title
    seen: set[str] = set()
    deduped = []
    for r in sorted(rows, key=lambda x: x.get("magnitude") or 0, reverse=True):
        if r["title"] not in seen:
            seen.add(r["title"])
            deduped.append(r)
    return deduped[:8]


# ---------------------------------------------------------------------------
# Source 5 — yfinance  (fixed: use Ticker.history, not yf.download)
# ---------------------------------------------------------------------------

def get_financial_snapshot(ticker: str) -> dict:
    """
    yfinance gives a live market pulse without requiring a paid API key.
    FIX: yf.download() silently returns empty DataFrame in restricted envs
    (AMD network, Docker).  yf.Ticker().history() is more robust.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"ticker": None, "available": False}

    try:
        import yfinance as yf

        t    = yf.Ticker(ticker)
        hist = t.history(period="3mo", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            # Second attempt with a longer period in case of thin data
            hist = t.history(period="6mo", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return {"ticker": ticker, "available": False}

        close   = hist["Close"].dropna()
        returns = close.pct_change().dropna()
        last_close  = float(close.iloc[-1])
        prev_close  = float(close.iloc[-2]) if len(close) > 1 else last_close
        drawdown    = float((close / close.cummax() - 1.0).min()) if len(close) > 2 else 0.0

        info      = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        return {
            "ticker":          ticker,
            "available":       True,
            "last_close":      last_close,
            "prev_close":      prev_close,
            "price_change_5d": _pct_change(close, 5),
            "price_change_1m": _pct_change(close, 21),
            "price_change_3m": _pct_change(close, min(63, len(close) - 1)),
            "volatility_20d":  float(returns.tail(20).std() * math.sqrt(252)) if len(returns) > 5 else 0.0,
            "drawdown_3m":     drawdown,
            "company_name":    info.get("longName") or info.get("shortName") or ticker,
            "market_cap":      info.get("marketCap"),
            "sector":          info.get("sector") or "",
            "series":          [
                {"date": str(idx.date()), "close": float(val)}
                for idx, val in close.tail(60).items()
            ],
        }
    except Exception as exc:
        return {"ticker": ticker, "available": False, "error": str(exc)}


def _pct_change(close: pd.Series, window: int) -> float:
    """Use a simple trailing return so the dashboard stays explainable."""
    if len(close) <= window:
        return 0.0
    start = float(close.iloc[-window - 1])
    end   = float(close.iloc[-1])
    if start == 0:
        return 0.0
    return (end / start - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_live_query_bundle(profile: SupplierProfile) -> dict[str, list[str]]:
    """
    Build supplier-specific search strings for each risk lane.
    FIX: Removed parentheses from terms — GDELT v2 does not support them.
    Plain AND keyword phrasing works across all three search sources.
    """
    supplier = profile.name
    country  = profile.hq_country

    geo_terms = [
        "sanctions", "tariff", "export control", "trade restriction", "war",
        "conflict", "military exercise", "blockade", "port strike", "coup",
    ]
    esg_terms = [
        "labor violation", "factory fire", "spill", "pollution", "emissions",
        "child labor", "bribery", "audit finding", "human rights", "workplace safety",
    ]
    logistics_terms = [
        "earthquake", "flood", "typhoon", "hurricane", "cyclone", "storm",
        "port congestion", "shipping disruption", "rail disruption", "shipping delay",
    ]
    finance_terms = [
        "bankruptcy", "downgrade", "liquidity crisis", "missed earnings",
        "credit rating cut", "insolvency", "default", "guidance cut", "debt covenant breach",
    ]

    return {
        "geopolitical": [f"{supplier} {term}" for term in geo_terms],
        "esg":          [f"{supplier} {term}" for term in esg_terms],
        "logistics":    [f"{country} {term}"  for term in logistics_terms],
        "financial":    [f"{supplier} {term}" for term in finance_terms],
        "news": [
            f"{supplier} disruption OR risk OR supply OR shutdown OR delay OR strike OR sanction OR tariff",
            f"{country} sanction OR export control OR military OR unrest OR earthquake OR flood",
        ],
    }


# ---------------------------------------------------------------------------
# Multi-source fetch with fallback chain
# ---------------------------------------------------------------------------

def _fetch_with_fallback(query: str, max_records: int = 8) -> list[dict]:
    """
    Try GDELT first, fall back to Bing News RSS, then DuckDuckGo.
    This ensures at least one source returns data even if GDELT is rate-limited
    or returns 0 results (which is common for narrow queries).
    """
    results = search_gdelt(query, max_records=max_records)
    if results:
        return results

    time.sleep(_REQUEST_DELAY)
    results = search_bing_news_rss(query, max_records=max_records)
    if results:
        return results

    time.sleep(_REQUEST_DELAY)
    return search_ddg_news(query, max_records=max_records)


# ---------------------------------------------------------------------------
# Main pipeline: build typed RiskEvent records
# ---------------------------------------------------------------------------

def build_event_records(profile: SupplierProfile, bundle: dict[str, list[str]]) -> list[RiskEvent]:
    """
    Convert raw live search results into typed evidence records.
    Uses a 3-source fallback chain so the agents always have something to reason about.
    """
    records: list[RiskEvent] = []

    # ── Geopolitical lane ────────────────────────────────────────────────────
    geo_kw = ["sanction", "war", "conflict", "export", "strike", "blockade", "unrest", "tariff", "coup"]
    for query in bundle["geopolitical"][:3]:          # slightly more queries than original
        for item in _fetch_with_fallback(query, max_records=5):
            text = item["title"] + " " + item["snippet"]
            sev  = _score_from_text(text, geo_kw, base=55)
            records.append(RiskEvent(
                source=item["source"], category="geopolitical",
                title=item["title"],  url=item["url"],
                published_at=item["published_at"], snippet=item["snippet"],
                severity=sev, entities=profile.focus_terms[:4],
                location=profile.hq_country, raw_score=sev,
            ))
        time.sleep(_REQUEST_DELAY)

    # ── ESG lane ─────────────────────────────────────────────────────────────
    esg_kw = ["labor", "pollution", "fire", "spill", "child", "bribery", "human rights", "workplace", "emissions"]
    for query in bundle["esg"][:3]:
        for item in _fetch_with_fallback(query, max_records=5):
            text = item["title"] + " " + item["snippet"]
            sev  = _score_from_text(text, esg_kw, base=46)
            records.append(RiskEvent(
                source=item["source"], category="esg",
                title=item["title"],  url=item["url"],
                published_at=item["published_at"], snippet=item["snippet"],
                severity=sev, entities=profile.focus_terms[:4],
                location=profile.hq_country, raw_score=sev,
            ))
        time.sleep(_REQUEST_DELAY)

    # ── Logistics lane (news) ─────────────────────────────────────────────────
    log_kw = ["earthquake", "flood", "port", "storm", "typhoon", "shipping", "delay", "disruption", "hurricane"]
    for query in bundle["logistics"][:3]:
        for item in _fetch_with_fallback(query, max_records=5):
            text = item["title"] + " " + item["snippet"]
            sev  = _score_from_text(text, log_kw, base=50)
            records.append(RiskEvent(
                source=item["source"], category="logistics",
                title=item["title"],  url=item["url"],
                published_at=item["published_at"], snippet=item["snippet"],
                severity=sev, entities=profile.focus_terms[:4],
                location=profile.hq_country, raw_score=sev,
            ))
        time.sleep(_REQUEST_DELAY)

    # ── Logistics lane (USGS structured) ─────────────────────────────────────
    # Pull both significant-hour AND 4.5+ day feeds; hour feed is often empty.
    for quake in get_usgs_significant_quakes():
        sev = float(quake.get("magnitude") or 5.5) * 12.0
        records.append(RiskEvent(
            source=quake["source"], category="logistics",
            title=quake["title"],   url=quake["url"],
            published_at=quake["published_at"], snippet=quake["snippet"],
            severity=min(95.0, sev), entities=profile.focus_terms[:4],
            location=quake.get("place"), raw_score=sev,
        ))

    # ── Financial lane ───────────────────────────────────────────────────────
    fin_kw = ["bankruptcy", "downgrade", "liquidity", "default", "insolvency", "guidance", "debt", "earnings"]
    for query in bundle["financial"][:2]:
        for item in _fetch_with_fallback(query, max_records=4):
            text = item["title"] + " " + item["snippet"]
            sev  = _score_from_text(text, fin_kw, base=60)
            records.append(RiskEvent(
                source=item["source"], category="financial",
                title=item["title"],  url=item["url"],
                published_at=item["published_at"], snippet=item["snippet"],
                severity=sev, entities=profile.focus_terms[:4],
                location=profile.hq_country, raw_score=sev,
            ))
        time.sleep(_REQUEST_DELAY)

    # ── General news sweep ───────────────────────────────────────────────────
    gen_kw = ["risk", "supply", "disruption", "shutdown", "sanction", "delay", "tariff", "strike"]
    for query in bundle["news"][:1]:
        for item in _fetch_with_fallback(query, max_records=6):
            text = item["title"] + " " + item["snippet"]
            sev  = _score_from_text(text, gen_kw, base=42)
            # Auto-classify by keyword dominance
            cat = "geopolitical"
            low = text.lower()
            if any(k in low for k in ["earthquake", "flood", "typhoon", "port", "shipping"]):
                cat = "logistics"
            elif any(k in low for k in ["labor", "pollution", "fire", "spill", "bribery"]):
                cat = "esg"
            elif any(k in low for k in ["bankruptcy", "downgrade", "default", "liquidity"]):
                cat = "financial"
            records.append(RiskEvent(
                source=item["source"], category=cat,
                title=item["title"],  url=item["url"],
                published_at=item["published_at"], snippet=item["snippet"],
                severity=sev, entities=profile.focus_terms[:4],
                location=profile.hq_country, raw_score=sev,
            ))
        time.sleep(_REQUEST_DELAY)

    # ── Deduplicate and rank ──────────────────────────────────────────────────
    unique: dict[tuple[str, str], RiskEvent] = {}
    for ev in sorted(records, key=lambda x: x.severity, reverse=True):
        key = (ev.category, ev.title[:80])          # truncate key to catch near-dupes
        if key not in unique:
            unique[key] = ev
    return _compact_events(list(unique.values()), max_items=15, per_category=4)
