import json
import os
import sys
import uuid

# logfire must be configured before app module imports so spans from
# chunking/loaders/embedding are captured from the start.
import logfire
from app.config import settings

if settings.LOGFIRE_TOKEN:
    if settings.LOGFIRE_BASE_URL:
        logfire.configure(
            token=settings.LOGFIRE_TOKEN,
            service_name="enterprise-ingestion-service",
            advanced=logfire.AdvancedOptions(base_url=settings.LOGFIRE_BASE_URL),
        )
    else:
        logfire.configure(token=settings.LOGFIRE_TOKEN, service_name="enterprise-ingestion-service")
    # Capture Jina embedding/reranking HTTP calls as structured Logfire spans.
    logfire.instrument_requests()
    logfire.instrument_openai()

from app.ingestion.chunkers.splitter import chunk_text, enforce_max_chunk_size
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.chroma_client import delete_collection, get_or_create_collection
from app.services.retrieval.embedding import embed_texts

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def _index_chunks(chunks: list[str], filename: str, source_type: str) -> int:
    """Embed chunks and add them to the Chroma collection. Returns point count."""
    # Guard against legacy chunks that exceed embedding API input limits.
    chunks = enforce_max_chunk_size(chunks)
    with logfire.span("Vectorizing & Indexing"):
        collection = get_or_create_collection()
        embeddings = embed_texts(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[
                {"source": filename, "source_type": source_type} for _ in chunks
            ],
        )
        logfire.info(f"Indexed {len(ids)} points to Chroma from {filename}.")
        return len(ids)


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → save locally → embed → index in Chroma."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office

                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed and index in Chroma
            _index_chunks(chunks, filename, source_type)

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def _ensure_collection():
    """Make sure the Chroma collection exists (dimension is inferred on first add)."""
    collection = get_or_create_collection()
    logfire.info(f"Collection '{settings.CHROMA_COLLECTION}' is ready.")
    return collection


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Chroma collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):
        # Wipe collection if requested (Chroma infers dimensions on first add,
        # so there is no explicit create step needed).
        if wipe:
            with logfire.span("Wiping Collection"):
                delete_collection()
        _ensure_collection()

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = "true" if "true" in base_name else "noisy" if "noisy" in base_name else "general"
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = "true" if "true" in subdir.lower() else "noisy" if "noisy" in subdir.lower() else subdir
                process_directory(os.path.join(base_dir, subdir), source_type)


def reindex_processed_data(processed_dir: str = PROCESSED_DATA_DIR, wipe: bool = False):
    """
    Re-index chunks previously saved as JSON in processed_data/<source_type>/.

    Raw source files (PDFs, DOCX, ...) are not required: the already-chunked
    text is embedded and upserted into the Chroma collection directly.
    """
    with logfire.span("Reindexing processed_data", path=processed_dir):
        if wipe:
            with logfire.span("Wiping Collection"):
                delete_collection()
        _ensure_collection()

        total_points = 0
        for source_type in sorted(os.listdir(processed_dir)):
            folder = os.path.join(processed_dir, source_type)
            if not os.path.isdir(folder):
                continue
            for filename in sorted(os.listdir(folder)):
                if not filename.endswith(".json"):
                    continue
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                chunks = data.get("chunks", [])
                if not chunks:
                    continue
                total_points += _index_chunks(chunks, data.get("filename", filename), source_type)

        logfire.info(f"Reindex complete — {total_points} points in '{settings.CHROMA_COLLECTION}'.")


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    #   python -m app.ingestion.processor --reindex processed_data --wipe
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    reindex_requested = "--reindex" in clean_args
    target_args = [a for a in clean_args if a != "--reindex"]

    if reindex_requested:
        processed_dir = target_args[1] if len(target_args) > 1 else PROCESSED_DATA_DIR
        if not os.path.exists(processed_dir):
            print(f"Error: path '{processed_dir}' does not exist.")
            sys.exit(1)
        reindex_processed_data(processed_dir, wipe=wipe_requested)
        logfire.info("Reindex job completed.")
        sys.exit(0)

    target_dir = target_args[1] if len(target_args) > 1 else "DATA"
    explicit_type = target_args[2] if len(target_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completed.")