# Supplier Risk Intelligence Control Tower

This build is for the AMD hackathon procurement / supplier-risk track.

It is designed as a real-time control tower instead of a static scoring notebook.

- The brain runs on an open-source QWEN model served locally on the AMD GPU.
- CrewAI orchestrates the analyst agents.
- Gradio shows the live dashboard.
- The background work is visible while the monitor runs.
- Memory, graph memory, and runtime logs persist across runs.

## Pattern choice

The implementation uses:

- orchestrated workflow
- parallel-ish lane analysis
- generator / critic validation
- long-term memory
- graph memory
- human-readable operator dashboard

That fits supplier risk better than a single-agent chat flow because the problem is event-driven, multi-source, and continuous.

## Live signals

The system polls public live sources at runtime:

GDELT_ENDPOINT    = "https://api.gdeltproject.org/api/v2/doc/doc"
USGS_SIGNIFICANT  = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_hour.geojson"
USGS_4_5_DAY      = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
DDG_SEARCH        = "https://html.duckduckgo.com/html/"
GNEWS_RSS         = "https://gnews.io/api/v4/search"          # free tier, no key needed for RSS fallback
BING_NEWS_RSS     = "https://www.bing.com/news/search"

These are fetched live, not taken from canned sample rows.

## How the pipeline works

1. A supplier digital twin is created from the UI or notebook inputs.
2. Live queries are built for financial, geopolitical, ESG, and logistics risk.
3. The fetch layer collects live events.
4. The graph memory layer links supplier, country, and event entities.
5. The risk engine fuses the signals into a score and risk band.
6. CrewAI agents produce lane briefs, a summary, and a critic pass.
7. Gradio renders the result and exposes the runtime logs.

## Model setup on AMD GPU

VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --api-key abc-123 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

The endpoint is local. It is protocol-compatible so CrewAI can talk to it cleanly.

- `supplier_risk_intelligence_realtime.ipynb`

## Dashboard outputs

The Gradio UI shows:

- the latest risk snapshot
- live evidence table
- risk drift chart
- rolling memory
- graph memory summary
- runtime log
- persisted state
- crew lane outputs

## Notebook flow

The notebook is intentionally simple:

- configure the local model endpoint
- define a supplier twin
- run one assessment
- run continuous monitoring
- launch the Gradio dashboard

## Files

- `supplier_risk_intelligence_realtime.ipynb`
- `requirements.txt`
- `src/config.py`
- `src/live_sources.py`
- `src/memory_store.py`
- `src/graph_memory.py`
- `src/risk_engine.py`
- `src/llm_tools.py`
- `src/orchestration.py`
- `src/app.py`

## Notes

- `src/__init__.py` is intentionally empty.
- All comments are plain code comments only.
- The system is designed for the contained AMD environment with persistent shared storage.
