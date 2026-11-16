import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
def resolve_database_path(
    configured_path: str | None,
    *,
    root_dir: Path = ROOT_DIR,
) -> Path:
    """Resolve an optional SQLite override; defaults to data/sample/sales.db."""
    default_path = root_dir / "data" / "sample" / "sales.db"
    if configured_path is None or not configured_path.strip():
        return default_path

    candidate = Path(configured_path.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = root_dir / candidate
    return candidate.resolve()


DATABASE_PATH = resolve_database_path(os.getenv("BI_MULTIAGENT_DATABASE_PATH"))
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

MAX_HISTORY_MESSAGES = 16
MAX_SQL_ROWS = 100
POSTGRES_CHECKPOINTER_URI = os.getenv("POSTGRES_CHECKPOINTER_URI")
