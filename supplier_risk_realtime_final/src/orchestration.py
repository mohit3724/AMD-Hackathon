from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd

from .config import SupplierProfile
from .graph_memory import GraphMemory
from .live_sources import build_live_query_bundle, summarize_events_for_llm
from .memory_store import EventMemory, summarise_memory
from .risk_engine import build_monitor_bundle, run_risk_engine
from .llm_tools import make_agent, make_llm, run_critic_crew, run_lane_brief, run_summary_crew


SYSTEM_BACKBONE = """
You are the chief supplier-risk analyst for a procurement control tower.

Goals:
- Detect live supplier risk from financial, geopolitical, ESG, and logistics signals.
- Be concise, factual, and operational.
- Use only the evidence provided.
- Highlight immediate actions when risk is high.
- Never invent events or dates.
""".strip()


def _lane_agents(model_name: str, base_url: str | None = None, api_key: str | None = None):
    # Separate lane agents keep the trace easy to follow in the UI.
    _model = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
    llm = make_llm(model_name=_model, base_url=base_url, api_key=api_key)
    geo = make_agent(
        role="Geopolitical Risk Analyst",
        goal="Assess sanctions, conflict, trade control, and border risk.",
        backstory=SYSTEM_BACKBONE,
        llm=llm,
    )
    esg = make_agent(
        role="ESG Risk Analyst",
        goal="Assess labor, safety, pollution, and reputation risk.",
        backstory=SYSTEM_BACKBONE,
        llm=llm,
    )
    logi = make_agent(
        role="Logistics Risk Analyst",
        goal="Assess transport, weather, port, and physical disruption risk.",
        backstory=SYSTEM_BACKBONE,
        llm=llm,
    )
    fin = make_agent(
        role="Financial Risk Analyst",
        goal="Assess liquidity, volatility, leverage, and credit stress.",
        backstory=SYSTEM_BACKBONE,
        llm=llm,
    )
    synth = make_agent(
        role="Risk Synthesis Lead",
        goal="Fuse all lanes into a procurement action memo.",
        backstory=SYSTEM_BACKBONE,
        llm=llm,
    )
    critic = make_agent(
        role="Grounded Critic",
        goal="Check the memo for hallucination and missing action.",
        backstory=SYSTEM_BACKBONE,
        llm=llm,
    )
    return {"geo": geo, "esg": esg, "logi": logi, "fin": fin, "synth": synth, "critic": critic}



def _sort_recent_events(events):
    return sorted(
        events,
        key=lambda e: (getattr(e, "published_at", "") or "", float(getattr(e, "severity", 0.0) or 0.0)),
        reverse=True,
    )

def _lane_evidence(events, category: str, limit: int = 3):
    lane_events = [e for e in _sort_recent_events(events) if getattr(e, "category", None) == category]
    return lane_events[:limit]


async def assess_supplier(
    profile: SupplierProfile,
    memory: EventMemory,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    # Gather the live evidence before the crew starts reasoning.
    bundle = build_monitor_bundle(profile, memory)
    events = bundle["events"]
    fin = bundle["financial_snapshot"]
    queries = bundle["queries"]

    recent_events = _sort_recent_events(events)[:8]
    payload = {
        "supplier": profile.__dict__,
        "financial_snapshot": fin,
        "events": [e.as_dict() for e in recent_events],
        "events_summary": summarize_events_for_llm(events, max_items=8),
        "memory": summarise_memory(memory, limit=8),
        "category_queries": queries,
    }

    agents = _lane_agents(model_name=model_name, base_url=base_url, api_key=api_key)

    # Run each lane in sequence so the dashboard can show the intermediate work.
    geo_payload = {"supplier": profile.__dict__, "lane": "geopolitical", "evidence": [e.as_dict() for e in _lane_evidence(events, "geopolitical", 3)]}
    esg_payload = {"supplier": profile.__dict__, "lane": "esg", "evidence": [e.as_dict() for e in _lane_evidence(events, "esg", 3)]}
    logi_payload = {"supplier": profile.__dict__, "lane": "logistics", "evidence": [e.as_dict() for e in _lane_evidence(events, "logistics", 3)]}
    fin_payload = {"supplier": profile.__dict__, "lane": "financial", "evidence": [{"metric": k, "value": v} for k, v in fin.items()] if isinstance(fin, dict) else fin}


    # geo_brief = await asyncio.to_thread(run_lane_brief, agents["geo"], "geopolitical", geo_payload)
    # esg_brief = await asyncio.to_thread(run_lane_brief, agents["esg"], "esg", esg_payload)
    # logi_brief = await asyncio.to_thread(run_lane_brief, agents["logi"], "logistics", logi_payload)
    # fin_brief = await asyncio.to_thread(run_lane_brief, agents["fin"], "financial", fin_payload)
    (geo_brief, esg_brief, logi_brief, fin_brief) = await asyncio.gather(
    asyncio.to_thread(run_lane_brief, agents["geo"],  "geopolitical", geo_payload),
    asyncio.to_thread(run_lane_brief, agents["esg"],  "esg",          esg_payload),
    asyncio.to_thread(run_lane_brief, agents["logi"], "logistics",    logi_payload),
    asyncio.to_thread(run_lane_brief, agents["fin"],  "financial",    fin_payload),
    )
    
    payload["lane_briefs"] = {
        "geopolitical": geo_brief,
        "esg": esg_brief,
        "logistics": logi_brief,
        "financial": fin_brief,
    }

    summary = await asyncio.to_thread(run_summary_crew, [agents["synth"]], payload)
    critique = await asyncio.to_thread(run_critic_crew, agents["critic"], summary, payload)

    snapshot, next_state = run_risk_engine(
        profile,
        memory,
        events,
        fin,
        llm_summary=json.dumps(summary, ensure_ascii=False),
        llm_critique=json.dumps(critique, ensure_ascii=False),
    )
    snapshot.state_delta = next_state

    # Persist the event stream and the latest state for future runs.
    graph = GraphMemory(base_dir=memory.base.as_posix(), supplier_name=profile.name.lower().replace(" ", "_"))
    graph.update(profile, events)
    graph_tail = graph.summary(top_k=8)

    memory.append_events(events)
    memory.save_state(next_state)
    memory.append_log("cycle_complete", {"event_count": len(events), "overall_score": snapshot.overall_score})

    return {
        "snapshot": snapshot.as_dict(),
        "financial_snapshot": fin,
        "event_count": len(events),
        "lane_briefs": payload["lane_briefs"],
        "summary": summary,
        "critique": critique,
        "memory_tail": summarise_memory(memory, limit=8),
        "graph_tail": graph_tail,
        "queries": queries,
    }


async def continuous_monitor(
    profile: SupplierProfile,
    memory: EventMemory,
    cycles: int = 3,
    pause_seconds: int = 300,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    base_url: str | None = None,
    api_key: str | None = None,
):
    # Repeated polling is the easiest way to demonstrate live monitoring.
    results = []
    for _ in range(cycles):
        results.append(await assess_supplier(profile, memory, model_name=model_name, base_url=base_url, api_key=api_key))
        await asyncio.sleep(pause_seconds)
    return results
