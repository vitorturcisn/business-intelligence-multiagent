"""Inicializa as tabelas de checkpoint no PostgreSQL."""

from bi_multiagent.persistence.postgres import postgres_checkpointer


def main() -> None:
    with postgres_checkpointer(setup=True):
        pass

    print("Tabelas de checkpoint inicializadas/verificadas.")


if __name__ == "__main__":
    main()
