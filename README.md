# Vantage RAG

Kartik's agentic RAG chatbot — a production-grade agentic RAG system built with **LangGraph**, a **Portkey LLM Gateway** (with Groq fallback), **Chroma** vector search, **Jina AI** embedding/reranking (with local fallbacks), and **NeMo Guardrails**. It answers questions from an indexed documentation corpus via a planner → orchestrator → sub-agent pipeline with a fact-checking pass.

## Key Features

- **Agentic Flow**: LangGraph state machine — planner classifies intent, orchestrator routes to researcher/analyst/coder sub-agents, responder synthesizes, fact_checker verifies grounding before returning.
- **Guardrails**: NeMo Guardrails gate runs before the graph and blocks injection, jailbreak, and off-domain inputs (`GUARDRAILS_FAIL_OPEN` configurable).
- **LLM Gateway**: All LLM calls route through Portkey (`@<slug>/<model>`, default `openai/gpt-oss-20b` under the `policy` slug) with an automatic Groq fallback when Portkey is unreachable.
- **Hybrid Retrieval**: Chroma Cloud vector search fused with in-memory BM25 via RRF, then re-ranked with the **Jina Reranker v3** API.
- **Embeddings**: `jina-embeddings-v3` (1024-dim) via Jina API, with local `mixedbread-ai/mxbai-embed-large-v1` fallback.
- **Local Parsing**: PDF, HTML, TXT, DOCX, PPTX parsed on-device — no external OCR service.
- **Observability**: Trace nesting with **Pydantic Logfire** across every node.
- **State**: Durable Postgres checkpointer (LangGraph `PostgresSaver`) with in-memory `MemorySaver` fallback.
- **API**: Sync `/query` and Server-Sent-Events `/query/stream`, optional bearer-token auth, Redis-backed (or in-memory) rate limiting.
- **Evaluation**: RAGAS-powered suite (5 metrics) with a Streamlit demo app and a headless `evals/run_evals.py --metrics` script.

---

## Agent Flow

```mermaid
graph TD
    User((User)) --> API[FastAPI /query]
    API --> Gate{NeMo Guardrails}
    Gate -->|Blocked| User
    Gate -->|Pass| Planner[Planner]
    Planner --> Orchestrator[Orchestrator]
    Orchestrator -->|chat| Responder[Responder]
    Orchestrator -->|tool| Tool[ToolExecutor]
    Orchestrator -->|research / code| Researcher[Researcher]
    Researcher -->|code intent| Coder[Coder]
    Researcher -->|analysis intent| Analyst[Analyst]
    Tool --> Responder
    Coder --> Responder
    Analyst --> Responder
    Responder --> FC[Fact Checker]
    FC --> User
```

Nodes in `app/agents/graph.py`: `planner` → `orchestrator` → sub-agents (`researcher`, `analyst`, `coder`, `tool_executor`) → `responder` → `fact_checker` → end.

---

## Providers

| Layer | Primary | Fallback |
|---|---|---|
| LLM | Portkey gateway (`@policy/openai/gpt-oss-20b`) | Groq direct (`openai/gpt-oss-20b`) |
| Embeddings | Jina API `jina-embeddings-v3` | local `mxbai-embed-large-v1` |
| Reranker | Jina API `jina-reranker-v3` | rank by vector score |
| Vector store | Chroma Cloud (`enterprise_rag`, cosine) | — |

Model/routing settings live in `app/config.py` and `app/gateway/client.py`.

## Repo Layout

```
app/
  main.py                 FastAPI app: /query, /query/stream, /graph
  agents/graph.py         LangGraph agent pipeline + checkpointer
  gateway/                Portkey client + Groq fallback
  guardrails/             NeMo rails gate
  services/retrieval/     Chroma, hybrid (BM25+vector), ranking, embedding
data/data/                Indexed corpus (PDF/DOCX/HTML/TXT/PPTX)
evals/
  pipeline.py             Eval pipeline wrapper around /query
  metrics.py              RAGAS 5-metric scoring (gateway judge)
  build_golden.py         Golden dataset generator (LLM-drafted + reviewed)
  run_evals.py            Headless runner (--metrics) → report.json
  app.py                  Streamlit eval dashboard
ui/st_cloud_ui.py         Main chat UI
```

## Evaluation

```bash
# 1. (One-time) draft + review the golden dataset
uv run python -m evals.build_golden --per-source 6     # writes evals/golden_dataset.json
#    review `reference` vs `relevant_contexts`, then set "_draft": false

# 2. With the backend running (uvicorn on :8000):
uv run python -m evals.run_evals                # guardrails pass-rate report
uv run python -m evals.run_evals --metrics      # full RAGAS run (slow, serialized)
uv run streamlit run evals/app.py               # interactive eval dashboard
```

Metrics (`evals/metrics.py`): Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness — scored with a Portkey gateway judge and `sentence-transformers/all-MiniLM-L6-v2` embeddings. Output: `evals/report.json`.

## API

- `POST /query` — `{question, thread_id}` → sync JSON answer with `plan`, `paths`, `tools`, `fact_check`.
- `POST /query/stream` — same input, Server-Sent-Events of node-by-node progress.
- `GET /graph` — graph visualization (auth-gated when `API_KEY` is set).

## Quick Start

```bash
uv sync                      # Windows-managed venv (WSL: use .venv/Scripts/python.exe)
cp .env.example .env         # fill in the keys below
uv run uvicorn app.main:app --reload --port 8000
uv run streamlit run ui/st_cloud_ui.py
```

Required env: `PORTKEY_API_KEY`, `PORTKEY_GATEWAY_URL`, `PORTKEY_PRIMARY_MODEL`, `GROQ_API_KEY` (fallback), `JINA_API_KEY` (embeddings/reranker). Optional: `POSTGRES_URI` (durable state), `API_KEY` (bearer auth), `REDIS_URL` (distributed rate limits).