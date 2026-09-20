- uv run logfire auth  # for authentication and login for logfire
- data ingestion:      uv run python -m app.ingestion.processor DATA/data true --wipe

- run backend:         uvicorn app.main:app --reload --port 8000
- run frontend:        uv run streamlit run ui/st_cloud_ui.py
- check graph:         http://localhost:8000/graph
                       uv run python -c "from app.agents.graph import build_graph; from langgraph.checkpoint.memory import MemorySaver; print(build_graph(MemorySaver()).get_graph().draw_mermaid())"

- evals (backend must be running on :8000):
    uv run python -m evals.build_golden --per-source 6   # draft goldens → evals/golden_dataset.json (review, then _draft: false)
    uv run python -m evals.run_evals                     # guardrails pass-rate report
    uv run python -m evals.run_evals --metrics           # full RAGAS pass (slow, serialized)
    uv run streamlit run evals/app.py                    # eval dashboard

- smoke-test the stream endpoint from WSL:
    curl.exe -s -X POST http://localhost:8000/query/stream -H "Content-Type: application/json" --data "@body.json"

- Golden dataset — evals/golden_dataset.json finalized: 15 RAG samples (LLM-drafted from the 3 Chroma sources, every reference cross-checked - against its relevant_contexts) + 6 guardrails cases (3 block / 3 pass), _draft: false. Fixed two issues along the way: cp1252 crashes on - - emoji prints (builder now ASCII-only) and empty LLM responses (JSON-array extraction now retries once with an "ONLY JSON" hint).

- evaluation -  uv run python -m evals.run_evals --metrics
- live query and response checking 