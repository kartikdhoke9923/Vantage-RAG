- uv run logfire auth  # for authentication and loging in logfire and setups 
- then data ingestion  python -m app.ingestion.processor DATA/data true --wipe
- but because i used uv pacage installer so i did uv run python -m app.ingestion.processor DATA/data true --wipe 

- now for running and checking frontend plus backend 
- uvicorn app.main:app --reload --port 8000

- uv run python -c "from app.agents.graph import build_graph; from langgraph.checkpoint.memory import MemorySaver; print(build_graph(MemorySaver()).get_graph().draw_mermaid())"
- use this for checking graphs or http://localhost:8000/graph 

- uv run streamlit run ui/st_cloud_ui.py   # to run frontend