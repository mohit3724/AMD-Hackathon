from __future__ import annotations

from typing import Any

from .config import RiskEvent, RiskSnapshot, SupplierProfile, now_iso
from .live_sources import build_live_query_bundle, get_financial_snapshot, build_event_records
from .memory_store import EventMemory, update_state_with_events


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


FINANCE_WEIGHTS = {
    "price_change_5d": -0.35,
    "price_change_1m": -0.22,
    "price_change_3m": -0.12,
    "volatility_20d": 18.0,
    "drawdown_3m": -55.0,
}

CATEGORY_BASE = {
    "financial": 35.0,
    "geopolitical": 42.0,
    "esg": 28.0,
    "logistics": 32.0,
}


def finance_risk_from_snapshot(snapshot: dict) -> tuple[float, list[str]]:
    # Convert live market movement into an interpretable risk score.
    if not snapshot.get("available"):
        return 20.0, ["No market data available."]

    score = 15.0
    drivers: list[str] = []
    for k, w in FINANCE_WEIGHTS.items():
        v = float(snapshot.get(k, 0.0) or 0.0)
        score += v * w if k.startswith("drawdown") or k.startswith("price_change") else min(35.0, v * w)
        if abs(v) > 0.01:
            drivers.append(f"{k}={v:.2f}")
    if snapshot.get("price_change_5d", 0) < -5:
        drivers.append("5D price selloff")
        score += 8
    if snapshot.get("drawdown_3m", 0) < -15:
        drivers.append("deep drawdown")
        score += 12
    if snapshot.get("volatility_20d", 0) > 0.4:
        drivers.append("elevated volatility")
        score += 10
    return _clip(score), drivers[:5]


def event_risk_score(events: list[RiskEvent], category: str) -> tuple[float, list[str], float]:
    # Cluster recent live events into one lane score and lane confidence.
    subset = [e for e in events if e.category == category]
    if not subset:
        return CATEGORY_BASE[category] * 0.45, [f"No fresh {category} signals."], 0.2

    max_sev = max(e.severity for e in subset)
    avg_sev = sum(e.severity for e in subset) / len(subset)
    recency_bonus = min(10.0, len(subset) * 1.5)
    concentration = min(12.0, len({e.title for e in subset[:10]}) * 1.2)
    score = CATEGORY_BASE[category] + 0.42 * max_sev + 0.24 * avg_sev + recency_bonus + concentration
    drivers = [f"{len(subset)} recent {category} events", f"peak severity {max_sev:.1f}"]
    top_titles = [e.title for e in sorted(subset, key=lambda x: x.severity, reverse=True)[:3]]
    drivers.extend(top_titles)
    confidence = min(1.0, 0.35 + 0.12 * len(subset))
    return _clip(score), drivers[:5], confidence


def fuse_scores(fin: float, geo: float, esg: float, logi: float, event_count: int) -> float:
    # Blend the lanes, then add a small novelty bump for fresh activity.
    raw = fin * 0.30 + geo * 0.35 + esg * 0.15 + logi * 0.20
    novelty = min(10.0, event_count * 0.35)
    return _clip(raw + novelty)


def band_from_score(score: float) -> str:
    if score >= 80:
        return "critical"
    if score >= 65:
        return "high"
    if score >= 45:
        return "moderate"
    return "low"


def recommend_actions(snapshot: RiskSnapshot, profile: SupplierProfile) -> list[str]:
    # Map scores to procurement actions instead of generic advice.
    actions: list[str] = []
    if snapshot.geopolitical_score >= 70:
        actions.append("Activate alternate sourcing for exposed lanes.")
    if snapshot.financial_score >= 70:
        actions.append("Recheck supplier liquidity, debt, and coverage ratios.")
    if snapshot.logistics_score >= 65:
        actions.append("Increase safety stock for critical SKUs.")
    if snapshot.esg_score >= 65:
        actions.append("Request an immediate ESG remediation attestation.")
    if snapshot.overall_score >= 80:
        actions.append("Escalate to procurement leadership for manual review.")
    if not actions:
        actions.append("Keep monitoring; no immediate intervention required.")
    return actions[:5]


def top_drivers_from_scores(category_scores: dict[str, float], drivers: dict[str, list[str]], events: list[RiskEvent]) -> list[str]:
    # Surface the strongest lane and the strongest evidence items.
    ranked = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    out: list[str] = []
    for cat, score in ranked[:2]:
        out.append(f"{cat}: {score:.1f}")
        out.extend(drivers.get(cat, [])[:2])
    if events:
        top_events = sorted(events, key=lambda e: e.severity, reverse=True)[:3]
        out.extend([f"{ev.category.upper()} | {ev.title}" for ev in top_events])
    return out[:8]


def run_risk_engine(
    profile: SupplierProfile,
    memory: EventMemory,
    events: list[RiskEvent],
    financial_snapshot: dict,
    llm_summary: str = "",
    llm_critique: str = "",
) -> tuple[RiskSnapshot, dict]:
    # Combine all live signals into a single supplier risk snapshot.
    fin_score, fin_drivers = finance_risk_from_snapshot(financial_snapshot)
    geo_score, geo_drivers, geo_conf = event_risk_score(events, "geopolitical")
    esg_score, esg_drivers, esg_conf = event_risk_score(events, "esg")
    log_score, log_drivers, log_conf = event_risk_score(events, "logistics")

    category_scores = {
        "financial": fin_score,
        "geopolitical": geo_score,
        "esg": esg_score,
        "logistics": log_score,
    }
    drivers = {
        "financial": fin_drivers,
        "geopolitical": geo_drivers,
        "esg": esg_drivers,
        "logistics": log_drivers,
    }

    overall = fuse_scores(fin_score, geo_score, esg_score, log_score, len(events))
    confidence = _clip((0.35 + geo_conf + esg_conf + log_conf) / 4 + (0.2 if financial_snapshot.get("available") else 0.05), 0, 1)

    snapshot = RiskSnapshot(
        supplier=profile.name,
        generated_at=now_iso(),
        financial_score=_clip(fin_score),
        geopolitical_score=_clip(geo_score),
        esg_score=_clip(esg_score),
        logistics_score=_clip(log_score),
        overall_score=_clip(overall),
        risk_band=band_from_score(overall),
        confidence=confidence,
        top_drivers=top_drivers_from_scores(category_scores, drivers, events),
        recommended_actions=[],
        evidence=[e.as_dict() for e in sorted(events, key=lambda e: e.severity, reverse=True)[:25]],
        llm_summary=llm_summary,
        llm_critique=llm_critique,
        state_delta={},
    )
    snapshot.recommended_actions = recommend_actions(snapshot, profile)

    previous = memory.load_state()
    next_state = update_state_with_events(previous, snapshot, events)
    next_state["supplier"] = profile.name
    next_state["ticker"] = profile.ticker
    next_state["financial_snapshot"] = financial_snapshot
    return snapshot, next_state


def build_monitor_bundle(profile: SupplierProfile, memory: EventMemory) -> dict[str, Any]:
    # Pull one consistent bundle of live queries, events, and market data.
    queries = build_live_query_bundle(profile)
    events = build_event_records(profile, queries)
    fin = get_financial_snapshot(profile.ticker or "")
    return {"queries": queries, "events": events, "financial_snapshot": fin}
