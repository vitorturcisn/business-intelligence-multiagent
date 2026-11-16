from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")


def main() -> int:
    uri = os.getenv("SUPABASE_DB_URI", "").strip()
    if not uri:
        print("SUPABASE_DB_URI ausente no .env.")
        return 2

    from bi_multiagent.retrieval.supabase_pgvector import vector_config

    config = vector_config()
    print("Conectando ao Supabase PostgreSQL (credenciais ocultas)...")
    try:
        with psycopg.connect(uri, connect_timeout=8) as conn:
            extension = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if not extension:
                print("Extensão pgvector não encontrada. Execute a migration SQL.")
                return 3

            relation = conn.execute(
                "SELECT to_regclass(%s)",
                (f"{config['schema']}.{config['table']}",),
            ).fetchone()[0]
            if not relation:
                print("Tabela vetorial não encontrada. Execute a migration SQL.")
                return 4

            count = conn.execute(
                f'SELECT COUNT(*) FROM "{config["schema"]}"."{config["table"]}"'
            ).fetchone()[0]
            print("Conexão PostgreSQL: OK")
            print("pgvector version:", extension[0])
            print("Tabela:", f"{config['schema']}.{config['table']}")
            print("Dimensão configurada:", config["dimension"])
            print("Documentos indexados:", count)
            print("SUPABASE PGVECTOR CHECK: OK")
            return 0
    except Exception as exc:  # noqa: BLE001
        print("SUPABASE PGVECTOR CHECK: FALHOU")
        print(type(exc).__name__ + ":", str(exc).replace(uri, "[URI OCULTADA]"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
