from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")


def split_text(text: str, *, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    """Paragraph-aware deterministic splitter; avoids another runtime dependency."""
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(0, end - overlap)
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index Markdown business knowledge into Supabase PGVector."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace all rows in the configured vector table.",
    )
    parser.add_argument("--source-dir", default=str(ROOT / "data" / "knowledge_base"))
    args = parser.parse_args()

    uri = os.getenv("SUPABASE_DB_URI", "").strip()
    if not uri:
        print("SUPABASE_DB_URI ausente no .env.")
        return 2

    from bi_multiagent.retrieval.supabase_pgvector import (
        configure_psycopg_event_loop,
        get_vector_store,
        vector_config,
    )

    # Psycopg documents that Windows' ProactorEventLoop is incompatible with
    # async connections. Configure SelectorEventLoop before PGEngine is built.
    changed_policy = configure_psycopg_event_loop()
    if sys.platform == "win32":
        print(
            "Windows asyncio policy:",
            "SelectorEventLoop configurado" if changed_policy else "SelectorEventLoop já ativo",
        )

    config = vector_config()
    source_dir = Path(args.source_dir).resolve()
    files = sorted(source_dir.glob("*.md"))
    if not files:
        print(f"Nenhum Markdown encontrado em {source_dir}")
        return 3

    documents: list[Document] = []
    ids: list[str] = []
    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        for chunk_index, chunk in enumerate(split_text(content)):
            digest = hashlib.sha256(
                f"{file_path.name}\0{chunk_index}\0{chunk}".encode("utf-8")
            ).hexdigest()
            ids.append(digest)
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path.name,
                        "source_path": file_path.relative_to(ROOT).as_posix()
                        if file_path.is_relative_to(ROOT)
                        else file_path.name,
                        "chunk_index": chunk_index,
                        "document_type": "business_knowledge",
                    },
                )
            )

    qualified = f'"{config["schema"]}"."{config["table"]}"'
    try:
        with psycopg.connect(uri, connect_timeout=8) as conn:
            current_count = conn.execute(
                f"SELECT COUNT(*) FROM {qualified}"
            ).fetchone()[0]
            if current_count and not args.replace:
                print(
                    f"A tabela já contém {current_count} registros. "
                    "Para substituir explicitamente o conteúdo, execute novamente com --replace."
                )
                return 4
            if args.replace:
                conn.execute(f"TRUNCATE TABLE {qualified}")

        store = get_vector_store()
        store.add_documents(documents=documents, ids=ids)
        print("SUPABASE KNOWLEDGE BASE LOAD: OK")
        print("Arquivos Markdown:", len(files))
        print("Chunks preparados:", len(documents))
        print("Tabela:", f"{config['schema']}.{config['table']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print("SUPABASE KNOWLEDGE BASE LOAD: FALHOU")
        print(type(exc).__name__ + ":", str(exc).replace(uri, "[URI OCULTADA]"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
