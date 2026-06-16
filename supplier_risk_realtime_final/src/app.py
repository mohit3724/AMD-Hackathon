from __future__ import annotations

import asyncio
import json

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from src.config import SupplierProfile
from src.memory_store import EventMemory
from src.orchestration import assess_supplier


def default_supplier() -> SupplierProfile:
    # The demo starts with a semiconductor supplier because it is highly exposed.
    return SupplierProfile(
        name="TSMC",
        ticker="TSM",
        hq_country="Taiwan",
        operating_regions=["Taiwan", "Japan", "United States"],
        categories=["semiconductor manufacturing", "foundry", "advanced nodes"],
        critical_inputs=["EUV", "specialty chemicals", "advanced packaging"],
        notes="High exposure to geopolitics, logistics, and advanced-node supply continuity.",
    )


def _memory(base_dir: str, supplier_name: str) -> EventMemory:
    # Keep per-supplier persistence isolated in shared storage.
    safe = supplier_name.lower().replace(" ", "_")
    return EventMemory(base_dir=base_dir, supplier_name=safe)


def _risk_trend_figure(history: list[dict]):
    # Show score drift so the user can see background learning over time.
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if history:
        xs = list(range(1, len(history) + 1))
        ys = [float(x.get("overall_score", 0)) for x in history]
        ax.plot(xs, ys, marker="o")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Run")
        ax.set_ylabel("Overall risk")
        ax.set_title("Risk drift across runs")
    else:
        ax.text(0.5, 0.5, "No history yet", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def _events_table(snapshot: dict):
    # Flatten the evidence list for a readable dashboard table.
    evidence = snapshot.get("evidence", []) or []
    if not evidence:
        return pd.DataFrame(columns=["published_at", "category", "severity", "title", "source"])
    return pd.DataFrame([
        {
            "published_at": ev.get("published_at", ""),
            "category": ev.get("category", ""),
            "severity": ev.get("severity", 0),
            "title": ev.get("title", ""),
            "source": ev.get("source", ""),
            "location": ev.get("location", ""),
        }
        for ev in evidence
    ])


def _runtime_log_text(memory: EventMemory) -> str:
    # Surface the execution trace so the background never feels hidden.
    logs = memory.load_logs(limit=50)
    if not logs:
        return "No runtime logs yet."
    lines = []
    for row in logs[-20:]:
        lines.append(
            f"{row.get('timestamp','')} | {row.get('message','')} | "
            f"{json.dumps(row.get('payload', {}), ensure_ascii=False)}"
        )
    return "\n".join(lines)


def _state_summary(memory: EventMemory) -> str:
    # Summarize the latest persisted state for a quick operator view.
    state = memory.load_state()
    if not state:
        return "No persisted state yet."
    return json.dumps(
        {
            "last_updated": state.get("last_updated"),
            "latest_scores": state.get("latest_scores", {}),
            "category_counts": state.get("category_counts", {}),
            "history_points": len(state.get("history", [])),
        },
        indent=2,
        ensure_ascii=False,
    )


async def run_once(
    supplier_name: str,
    ticker: str,
    country: str,
    regions: str,
    categories: str,
    critical_inputs: str,
    notes: str,
    model_name: str,
    base_url: str,
    api_key: str,
):
    # Build the supplier twin from the UI inputs.
    supplier = SupplierProfile(
        name=supplier_name.strip() or "Unnamed Supplier",
        ticker=ticker.strip() or None,
        hq_country=country.strip() or "Unknown",
        operating_regions=[x.strip() for x in regions.split(",") if x.strip()],
        categories=[x.strip() for x in categories.split(",") if x.strip()],
        critical_inputs=[x.strip() for x in critical_inputs.split(",") if x.strip()],
        notes=notes.strip(),
    )
    memory = _memory("shared", supplier.name)
    memory.append_log("run_started", {"supplier": supplier.name, "ticker": supplier.ticker})

    result = await assess_supplier(
        supplier,
        memory,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )
    snapshot = result["snapshot"]
    history = memory.load_state().get("history", [])

    return (
        snapshot,
        _events_table(snapshot),
        _risk_trend_figure(history),
        result["memory_tail"],
        "\n".join(result["graph_tail"]),
        _runtime_log_text(memory),
        _state_summary(memory),
        json.dumps(result["lane_briefs"], indent=2, ensure_ascii=False),
    )


async def stream_monitor(
    supplier_name: str,
    ticker: str,
    country: str,
    regions: str,
    categories: str,
    critical_inputs: str,
    notes: str,
    model_name: str,
    base_url: str,
    api_key: str,
    cycles: int,
    pause_seconds: int,
):
    # Stream updates after each cycle so the user can watch the pipeline work.
    supplier = SupplierProfile(
        name=supplier_name.strip() or "Unnamed Supplier",
        ticker=ticker.strip() or None,
        hq_country=country.strip() or "Unknown",
        operating_regions=[x.strip() for x in regions.split(",") if x.strip()],
        categories=[x.strip() for x in categories.split(",") if x.strip()],
        critical_inputs=[x.strip() for x in critical_inputs.split(",") if x.strip()],
        notes=notes.strip(),
    )
    memory = _memory("shared", supplier.name)
    memory.append_log("monitor_started", {"supplier": supplier.name, "cycles": cycles, "pause_seconds": pause_seconds})

    last_result = None
    for idx in range(cycles):
        memory.append_log("cycle_start", {"cycle": idx + 1})
        result = await assess_supplier(
            supplier,
            memory,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )
        last_result = result
        snapshot = result["snapshot"]
        history = memory.load_state().get("history", [])
        status = f"Cycle {idx + 1}/{cycles} complete | risk={snapshot['overall_score']:.1f} | band={snapshot['risk_band']}"

        yield (
            snapshot,
            _events_table(snapshot),
            _risk_trend_figure(history),
            result["memory_tail"],
            "\n".join(result["graph_tail"]),
            _runtime_log_text(memory),
            _state_summary(memory),
            json.dumps(result["lane_briefs"], indent=2, ensure_ascii=False),
            status,
        )

        if idx < cycles - 1:
            memory.append_log("cycle_sleep", {"seconds": pause_seconds})
            await asyncio.sleep(pause_seconds)

    if last_result is None:
        last_result = {"snapshot": {}, "memory_tail": "", "graph_tail": [], "lane_briefs": {}}

    yield (
        last_result["snapshot"],
        _events_table(last_result["snapshot"]),
        _risk_trend_figure(memory.load_state().get("history", [])),
        last_result.get("memory_tail", ""),
        "\n".join(last_result.get("graph_tail", [])),
        _runtime_log_text(memory),
        _state_summary(memory),
        json.dumps(last_result.get("lane_briefs", {}), indent=2, ensure_ascii=False),
        "Monitoring finished.",
    )


def build_ui():
    # The dashboard keeps both the outputs and the background trail visible.
    with gr.Blocks(title="Supplier Risk Intelligence Control Tower") as demo:
        gr.Markdown("# Supplier Risk Intelligence Control Tower")
        gr.Markdown("Open-source model on AMD GPU, CrewAI lanes, live signals, and visible runtime trace.")

        with gr.Row():
            with gr.Column(scale=1):
                supplier_name = gr.Textbox(value="TSMC", label="Supplier name")
                ticker = gr.Textbox(value="TSM", label="Ticker")
                country = gr.Textbox(value="Taiwan", label="HQ country")
                regions = gr.Textbox(value="Taiwan, Japan, United States", label="Operating regions")
                categories = gr.Textbox(value="semiconductor manufacturing, foundry, advanced nodes", label="Categories")
                critical_inputs = gr.Textbox(value="EUV, specialty chemicals, advanced packaging", label="Critical inputs")
                notes = gr.Textbox(value="High exposure to geopolitics and logistics.", label="Notes", lines=3)

                model_name = gr.Textbox(value="Qwen/Qwen2.5-7B-Instruct", label="HF model served on AMD GPU")
                base_url = gr.Textbox(value="http://localhost:8000/v1", label="Local model base URL")
                api_key = gr.Textbox(value="abc-123", label="Local API token", type="password")
                cycles = gr.Slider(1, 10, value=1, step=1, label="Monitoring cycles")
                pause_seconds = gr.Slider(1, 600, value=5, step=1, label="Pause between cycles (seconds)")

                run_btn = gr.Button("Run once", variant="primary")
                stream_btn = gr.Button("Start monitor", variant="secondary")

            with gr.Column(scale=1.4):
                status = gr.Textbox(label="Status", value="Idle", lines=1)
                snapshot = gr.JSON(label="Risk snapshot")
                events = gr.Dataframe(label="Live evidence", interactive=False)
                trend = gr.Plot(label="Risk drift")
                memory_tail = gr.Textbox(label="Rolling memory", lines=8)
                graph_tail = gr.Textbox(label="Graph memory summary", lines=8)
                runtime_log = gr.Textbox(label="Runtime log", lines=12)
                state_view = gr.Textbox(label="Persisted state", lines=8)
                lane_briefs = gr.Textbox(label="Crew outputs", lines=18)

        run_btn.click(
            fn=run_once,
            inputs=[supplier_name, ticker, country, regions, categories, critical_inputs, notes, model_name, base_url, api_key],
            outputs=[snapshot, events, trend, memory_tail, graph_tail, runtime_log, state_view, lane_briefs],
        )

        stream_btn.click(
            fn=stream_monitor,
            inputs=[supplier_name, ticker, country, regions, categories, critical_inputs, notes, model_name, base_url, api_key, cycles, pause_seconds],
            outputs=[snapshot, events, trend, memory_tail, graph_tail, runtime_log, state_view, lane_briefs, status],
        )

    return demo


def launch():
    # Launch in notebook or as a script, depending on how the user runs it.
    demo = build_ui()
    demo.queue(default_concurrency_limit=4)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    launch()
