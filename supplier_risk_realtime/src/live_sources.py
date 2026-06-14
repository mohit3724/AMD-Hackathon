from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from urllib.parse import quote_plus

import pandas as pd
import requests

from .config import RiskEvent, SupplierProfile, now_iso

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
USGS_SIGNIFICANT_HOUR = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_hour.geojson"


def _safe_ts(value: str | None) -> str:
    # Convert heterogeneous timestamps into a single UTC ISO format.
    if not value:
        return now_iso()
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat()
    except Exception:
        return now_iso()


def _clean_text(text: str, limit: int = 240) -> str:
    # Strip noise and trim the text for compact evidence cards.
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def _score_from_text(text: str, keywords: list[str], base: float = 40.0, cap: float = 95.0) -> float:
    # Heuristic severity before the risk engine fuses everything.
    lower = (text or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    if hits == 0:
        return base * 0.35
    return min(cap, base + 12 * hits)


def search_gdelt(query: str, max_records: int = 12) -> list[dict]:
    # GDELT provides live global event discovery without a cooked dataset.
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "HybridRel",
    }
    try:
        resp = requests.get(GDELT_ENDPOINT, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    articles = payload.get("articles", []) or []
    rows = []
    for art in articles:
        rows.append({
            "source": "GDELT",
            "title": art.get("title") or art.get("seendate") or "Untitled",
            "url": art.get("url") or "",
            "published_at": _safe_ts(art.get("seendate")),
            "snippet": _clean_text(art.get("snippet") or art.get("title") or ""),
            "domain": art.get("domain") or "",
            "language": art.get("language") or "",
            "sourcecountry": art.get("sourcecountry") or "",
        })
    return rows


def get_usgs_significant_quakes() -> list[dict]:
    # Earthquakes often create logistics risk that procurement teams miss.
    try:
        resp = requests.get(USGS_SIGNIFICANT_HOUR, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    rows = []
    for feat in payload.get("features", []) or []:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None, None])
        rows.append({
            "source": "USGS",
            "title": props.get("title") or "Earthquake",
            "url": props.get("url") or "",
            "published_at": _safe_ts(
                props.get("time") and datetime.fromtimestamp(props["time"] / 1000, tz=timezone.utc).isoformat()
            ),
            "snippet": _clean_text(props.get("place") or props.get("title") or ""),
            "magnitude": props.get("mag"),
            "place": props.get("place"),
            "latitude": coords[1],
            "longitude": coords[0],
            "depth_km": coords[2],
            "alert": props.get("alert"),
        })
    return rows


def get_financial_snapshot(ticker: str) -> dict:
    # yfinance gives a live market pulse without requiring a paid API key.
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {"ticker": None, "available": False}

    try:
        import yfinance as yf

        hist = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if hist is None or hist.empty:
            return {"ticker": ticker, "available": False}

        close = hist["Close"].dropna()
        returns = close.pct_change().dropna()
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close
        drawdown = float((close / close.cummax() - 1.0).min()) if len(close) > 2 else 0.0

        return {
            "ticker": ticker,
            "available": True,
            "last_close": last_close,
            "prev_close": prev_close,
            "price_change_5d": _pct_change(close, 5),
            "price_change_1m": _pct_change(close, 21),
            "price_change_3m": _pct_change(close, min(63, len(close) - 1)),
            "volatility_20d": float(returns.tail(20).std() * math.sqrt(252)) if len(returns) > 5 else 0.0,
            "drawdown_3m": drawdown,
            "series": [{"date": str(idx.date()), "close": float(val)} for idx, val in close.tail(60).items()],
        }
    except Exception:
        return {"ticker": ticker, "available": False}


def _pct_change(close: pd.Series, window: int) -> float:
    # Use a simple trailing return so the dashboard stays explainable.
    if len(close) <= window:
        return 0.0
    start = float(close.iloc[-window - 1])
    end = float(close.iloc[-1])
    if start == 0:
        return 0.0
    return (end / start - 1.0) * 100.0


def build_live_query_bundle(profile: SupplierProfile) -> dict[str, list[str]]:
    # Build supplier-specific search strings for each risk lane.
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
        "bankruptcy", "downgrade", "liquidity", "missed earnings", "credit rating",
        "insolvency", "default", "guidance cut", "debt covenant",
    ]

    supplier_anchor = profile.name
    country_anchor = profile.hq_country
    return {
        "geopolitical": [f'"{supplier_anchor}" ({term})' for term in geo_terms],
        "esg": [f'"{supplier_anchor}" ({term})' for term in esg_terms],
        "logistics": [f'"{country_anchor}" ({term})' for term in logistics_terms],
        "financial": [f'"{supplier_anchor}" ({term})' for term in finance_terms],
        "news": [
            f'"{supplier_anchor}" (disruption OR risk OR supply OR shutdown OR delay OR strike OR sanction OR tariff)',
            f'"{country_anchor}" (sanction OR export control OR military OR unrest OR earthquake OR flood)',
        ],
    }


def build_event_records(profile: SupplierProfile, bundle: dict[str, list[str]]) -> list[RiskEvent]:
    # Convert raw live search results into typed evidence records.
    records: list[RiskEvent] = []

    for query in bundle["geopolitical"][:4]:
        for item in search_gdelt(query, max_records=8):
            title = item["title"] + " " + item["snippet"]
            sev = _score_from_text(title, ["sanction", "war", "conflict", "export", "strike", "blockade", "unrest"], base=55)
            records.append(RiskEvent(
                source=item["source"],
                category="geopolitical",
                title=item["title"],
                url=item["url"],
                published_at=item["published_at"],
                snippet=item["snippet"],
                severity=sev,
                entities=profile.focus_terms[:4],
                location=profile.hq_country,
                raw_score=sev,
            ))

    for query in bundle["esg"][:4]:
        for item in search_gdelt(query, max_records=8):
            title = item["title"] + " " + item["snippet"]
            sev = _score_from_text(title, ["labor", "pollution", "fire", "spill", "child", "bribery", "human rights"], base=46)
            records.append(RiskEvent(
                source=item["source"],
                category="esg",
                title=item["title"],
                url=item["url"],
                published_at=item["published_at"],
                snippet=item["snippet"],
                severity=sev,
                entities=profile.focus_terms[:4],
                location=profile.hq_country,
                raw_score=sev,
            ))

    for query in bundle["logistics"][:4]:
        for item in search_gdelt(query, max_records=8):
            title = item["title"] + " " + item["snippet"]
            sev = _score_from_text(title, ["earthquake", "flood", "port", "storm", "typhoon", "shipping", "delay"], base=50)
            records.append(RiskEvent(
                source=item["source"],
                category="logistics",
                title=item["title"],
                url=item["url"],
                published_at=item["published_at"],
                snippet=item["snippet"],
                severity=sev,
                entities=profile.focus_terms[:4],
                location=profile.hq_country,
                raw_score=sev,
            ))

    # Add structured logistics events from USGS so the feed is not news-only.
    for quake in get_usgs_significant_quakes():
        sev = float(quake.get("magnitude") or 5.5) * 12.0
        records.append(RiskEvent(
            source=quake["source"],
            category="logistics",
            title=quake["title"],
            url=quake["url"],
            published_at=quake["published_at"],
            snippet=quake["snippet"],
            severity=min(95.0, sev),
            entities=profile.focus_terms[:4],
            location=quake.get("place"),
            raw_score=sev,
        ))

    # Keep the event list compact and focused on the strongest signals.
    unique: dict[tuple[str, str], RiskEvent] = {}
    for ev in sorted(records, key=lambda x: x.severity, reverse=True):
        key = (ev.category, ev.title)
        if key not in unique:
            unique[key] = ev
    return list(unique.values())[:40]
