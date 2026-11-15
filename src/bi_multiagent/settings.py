import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
DATABASE_PATH = DATA_DIR / "sample" / "sales.db"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

MAX_HISTORY_MESSAGES = 16
MAX_SQL_ROWS = 100
POSTGRES_CHECKPOINTER_URI = os.getenv("POSTGRES_CHECKPOINTER_URI")
