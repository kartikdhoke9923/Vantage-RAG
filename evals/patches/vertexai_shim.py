"""Legacy backport shim for ragas 0.3.x.

langchain-community >=0.4 removed `langchain_community.chat_models.vertexai`;
ragas 0.3.2 still imports ChatVertexAI from there. Re-export from the modern
langchain-google-vertexai package.

This copy is the source of truth: after a `uv sync` that reinstalls
langchain-community, copy it back to
`.venv/Lib/site-packages/langchain_community/chat_models/vertexai.py`.
"""
from langchain_google_vertexai import ChatVertexAI, VertexAI, VertexAIEmbeddings

__all__ = ["ChatVertexAI", "VertexAI", "VertexAIEmbeddings"]