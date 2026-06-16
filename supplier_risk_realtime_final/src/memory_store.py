from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .config import RiskEvent, RiskSnapshot


class EventMemory:
    # Disk-backed memory keeps the control tower stateful across runs.
    def __init__(self, base_dir: str = "shared", supplier_name: str = "default") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / f"{supplier_name}_events.jsonl"
        self.state_path = self.base / f"{supplier_name}_state.json"
        self.log_path = self.base / f"{supplier_name}_runtime_log.jsonl"

    def append_events(self, events: Iterable[RiskEvent]) -> None:
        # Store each event as a JSON line so appends stay cheap.
        with self.path.open("a", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev.as_dict(), ensure_ascii=False) + "\n")

    def append_log(self, message: str, payload: dict | None = None) -> None:
        # Keep a visible execution trace for the Gradio dashboard.
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "payload": payload or {},
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load_logs(self, limit: int = 100) -> list[dict]:
        if not self.log_path.exists():
            return []
        rows = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        return rows[-limit:]

    def load_events(self, limit: int = 200) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        return rows[-limit:]

    def load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def query(self, text: str, limit: int = 8) -> list[dict]:
        # A tiny keyword retriever is enough for hackathon memory lookups.
        terms = [t.lower().strip() for t in text.split() if t.strip()]
        scored = []
        for ev in self.load_events(limit=500):
            hay = " ".join([
                str(ev.get("title", "")),
                str(ev.get("snippet", "")),
                str(ev.get("location", "")),
            ]).lower()
            hits = sum(1 for t in terms if t in hay)
            if hits:
                scored.append((hits, ev))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ev for _, ev in scored[:limit]]

    def recent_driver_summary(self, top_k: int = 8) -> list[tuple[str, int]]:
        # Surface which signal families are dominating the recent history.
        events = self.load_events(limit=400)
        counts = Counter(ev.get("category", "unknown") for ev in events)
        return counts.most_common(top_k)


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def decay_score(previous: float, delta: float, decay: float = 0.88) -> float:
    # Decay old state while still letting repeated events accumulate.
    return _clip(previous * decay + delta)


def update_state_with_events(prev_state: dict, snapshot: RiskSnapshot, events: list[RiskEvent]) -> dict:
    # Persist the latest state so future runs can see drift and recurrence.
    state = dict(prev_state or {})
    state.setdefault("history", [])
    state.setdefault("category_counts", {})
    state.setdefault("latest_scores", {})
    state.setdefault("last_updated", None)

    for ev in events:
        state["category_counts"][ev.category] = state["category_counts"].get(ev.category, 0) + 1

    state["latest_scores"] = {
        "financial": snapshot.financial_score,
        "geopolitical": snapshot.geopolitical_score,
        "esg": snapshot.esg_score,
        "logistics": snapshot.logistics_score,
        "overall": snapshot.overall_score,
    }
    state["last_updated"] = snapshot.generated_at

    history = state["history"][-49:]
    history.append({
        "generated_at": snapshot.generated_at,
        "overall_score": snapshot.overall_score,
        "risk_band": snapshot.risk_band,
        "top_drivers": snapshot.top_drivers[:4],
    })
    state["history"] = history
    return state


def summarise_memory(memory: EventMemory, limit: int = 12) -> str:
    # Give the model a compact rolling memory instead of the whole event log.
    events = memory.load_events(limit=150)
    if not events:
        return "No stored events yet."
    recent = events[-limit:]
    lines = []
    for ev in recent:
        lines.append(f'- [{ev.get("category", "")}] {ev.get("title", "")} ({ev.get("published_at", "")})')
    return "\n".join(lines)
