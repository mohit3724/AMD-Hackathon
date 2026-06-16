from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task

from .config import RiskSnapshot, SupplierProfile

def make_llm(model_name: str, base_url: str | None = None, api_key: str | None = None) -> LLM:
    # CrewAI uses LiteLLM internally; prefix tells it to use the OpenAI-compatible path
    _model = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
    return LLM(
        model=_model,
        base_url=base_url or os.environ.get("BASE_URL", "http://localhost:8000/v1"),
        api_key=api_key or os.environ.get("LOCAL_LLM_API_KEY", "abc-123"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        max_tokens=2048,           # ADD: prevent runaway generation
        timeout=120,               # ADD: 7B models on MI300X are fast but tasks are complex
    )


def make_agent(role: str, goal: str, backstory: str, llm: LLM, verbose: bool = True) -> Agent:
    # Each lane gets a focused agent so the analyst output stays legible.
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
    )


def _extract_json(text: str) -> dict[str, Any]:
    # The model may wrap JSON in prose; this keeps downstream code stable.
    if not text:
        return {}
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return {"text": text}


def run_lane_brief(agent: Agent, title: str, payload: dict[str, Any]) -> dict[str, Any]:
    # A one-task crew is enough to keep the behavior explicit and debuggable.
    task = Task(
        description=f"""
You are working on the {title} lane.

Use the following evidence to produce compact JSON with:
lane, severity, confidence, why_now, top_evidence, recommended_actions, watch_items.

Evidence:
{json.dumps(payload, indent=2, ensure_ascii=False)}
""".strip(),
        expected_output="Compact JSON only.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    output = crew.kickoff()
    return _extract_json(str(output))


def run_summary_crew(agents: list[Agent], payload: dict[str, Any]) -> dict[str, Any]:
    # The synthesis step merges lane-level evidence into an action memo.
    task = Task(
        description=f"""
Synthesize the full supplier-risk picture as JSON with:
summary, top_risks, why_now, recommended_actions, watch_items, confidence_note.

Do not invent facts beyond the evidence.

Evidence:
{json.dumps(payload, indent=2, ensure_ascii=False)}
""".strip(),
        expected_output="Compact JSON only.",
        agent=agents[0],
    )
    crew = Crew(agents=agents[:1], tasks=[task], process=Process.sequential, verbose=True)
    output = crew.kickoff()
    return _extract_json(str(output))


def run_critic_crew(agent: Agent, draft: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    # The critic checks grounding and missing operational advice.
    task = Task(
        description=f"""
Review this draft for unsupported claims, missing actions, and vague reasoning.

Draft:
{json.dumps(draft, indent=2, ensure_ascii=False)}

Evidence:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON with: pass, issues, fixes.
""".strip(),
        expected_output="Compact JSON only.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    output = crew.kickoff()
    return _extract_json(str(output))
