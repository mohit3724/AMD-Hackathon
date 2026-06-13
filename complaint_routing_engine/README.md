# Customer Complaint Classification & Routing Engine
### AMD TCS Hackathon | Agentic AI Solution

---

## Pattern: Single-Agent + FAISS RAG Memory + MCP + Generator-Critic

| Pattern | Role |
|---|---|
| Single-Agent Orchestration | perceive -> retrieve -> reason -> act -> persist -> reflect |
| RAG Memory (FAISS) | In-memory vector index over 8 SLA/routing policy docs |
| MCP Tool Registry | mcp-server-time for real-time SLA deadline computation |
| Generator-Critic Loop | Critic overrides low-confidence routing decisions |

---

## Setup

### 1. Launch vLLM (terminal):
```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-4B \
  --served-model-name Qwen3-4B \
  --api-key abc-123 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --max-model-len 8192
```

### 2. Install (notebook cell):
```bash
pip install pydantic-ai-slim openai faiss-cpu sentence-transformers mcp-server-time pandas
```

### 3. Run cells top to bottom.

---

## Why FAISS over ChromaDB

| | FAISS | ChromaDB |
|---|---|---|
| Install | faiss-cpu (single lightweight wheel) | Pulls FastAPI, Uvicorn, SQLite stack |
| Service | Fully in-process, no sidecar | Needs a running service |
| Persistence | Not needed for 8 policy docs | Overkill |
| AMD pod risk | Minimal | Higher chance of missing system deps |

---

## SLA Reference

| Category | Team | P0 | P1 | P2 | P3 |
|---|---|---|---|---|---|
| BILLING | Billing Team | 1h | 4h | 8h | 24h |
| TECHNICAL | Technical Support | 30min | 2h | 6h | 24h |
| ACCOUNT | Auth & Identity | 1h | 2h | 6h | 24h |
| FRAUD | Fraud & Risk | 15min | 1h | 4h | 12h |
| SHIPPING | Logistics | 2h | 4h | 8h | 48h |
| FEEDBACK | Product Team | -- | -- | -- | 72h |

