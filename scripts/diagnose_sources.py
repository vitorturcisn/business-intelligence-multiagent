import sqlite3
from pathlib import Path

from bi_multiagent.settings import DATABASE_PATH, KNOWLEDGE_BASE_DIR
from bi_multiagent.tools.business_tools import (
    get_sales_schema,
    search_business_knowledge,
)

print("\n" + "=" * 75)
print("1. SCHEMA REAL DO BANCO")
print("=" * 75)
print(get_sales_schema.func())

print("\n" + "=" * 75)
print("2. TABELAS DIRETAMENTE NO SQLITE")
print("=" * 75)

with sqlite3.connect(DATABASE_PATH) as connection:
    tables = connection.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    ).fetchall()

    for (table_name,) in tables:
        print(f"\nTabela: {table_name}")
        columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()

        for column in columns:
            print(f"  - {column[1]} ({column[2]})")

print("\n" + "=" * 75)
print("3. BUSCA DOCUMENTAL: INDICADORES")
print("=" * 75)
print(search_business_knowledge.func("indicadores de negócio"))

print("\n" + "=" * 75)
print("4. BUSCA DOCUMENTAL: RECEITA BRUTA E LÍQUIDA")
print("=" * 75)
print(search_business_knowledge.func("receita bruta receita líquida"))

print("\n" + "=" * 75)
print("5. CONTEÚDO INTEGRAL DA BASE DE CONHECIMENTO")
print("=" * 75)

for file_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
    print(f"\n\n### ARQUIVO: {file_path.name}")
    print(file_path.read_text(encoding="utf-8"))
    print("-" * 75)

print("\nDIAGNÓSTICO CONCLUÍDO.")
