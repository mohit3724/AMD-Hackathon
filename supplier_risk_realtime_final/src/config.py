from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

RiskCategory = Literal["financial", "geopolitical", "esg", "logistics"]


@dataclass
class SupplierProfile:
    # The supplier digital twin drives every query, score, and alert.
    name: str
    ticker: str | None
    hq_country: str
    operating_regions: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    critical_inputs: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def focus_terms(self) -> list[str]:
        # Keep all search anchors unique and clean.
        terms = [
            self.name,
            self.hq_country,
            *self.operating_regions,
            *self.categories,
            *self.critical_inputs,
        ]
        return [t for t in dict.fromkeys(x.strip() for x in terms if x and x.strip())]


@dataclass
class RiskEvent:
    # One live signal captured from a public source.
    source: str
    category: RiskCategory
    title: str
    url: str
    published_at: str
    snippet: str
    severity: float
    entities: list[str] = field(default_factory=list)
    location: str | None = None
    raw_score: float = 0.0

    def as_dict(self) -> dict:
        # JSON-safe form for memory, Gradio, and debugging.
        return {
            "source": self.source,
            "category": self.category,
            "title": self.title,
            "url": self.url,
            "published_at": self.published_at,
            "snippet": self.snippet,
            "severity": float(self.severity),
            "entities": self.entities,
            "location": self.location,
            "raw_score": float(self.raw_score),
        }


@dataclass
class RiskSnapshot:
    # One point-in-time risk view for the supplier.
    supplier: str
    generated_at: str
    financial_score: float
    geopolitical_score: float
    esg_score: float
    logistics_score: float
    overall_score: float
    risk_band: str
    confidence: float
    top_drivers: list[str]
    recommended_actions: list[str]
    evidence: list[dict]
    llm_summary: str
    llm_critique: str
    state_delta: dict

    def as_dict(self) -> dict:
        # Keep a single serialization path for dashboard and notebook output.
        return {
            "supplier": self.supplier,
            "generated_at": self.generated_at,
            "financial_score": self.financial_score,
            "geopolitical_score": self.geopolitical_score,
            "esg_score": self.esg_score,
            "logistics_score": self.logistics_score,
            "overall_score": self.overall_score,
            "risk_band": self.risk_band,
            "confidence": self.confidence,
            "top_drivers": self.top_drivers,
            "recommended_actions": self.recommended_actions,
            "evidence": self.evidence,
            "llm_summary": self.llm_summary,
            "llm_critique": self.llm_critique,
            "state_delta": self.state_delta,
        }


def now_iso() -> str:
    # Use UTC for stable event ordering across runs.
    return datetime.now(timezone.utc).isoformat()
