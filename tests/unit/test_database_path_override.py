from pathlib import Path

from bi_multiagent.settings import resolve_database_path


def test_database_path_defaults_to_project_sales_db(tmp_path):
    result = resolve_database_path("", root_dir=tmp_path)
    assert result == tmp_path / "data" / "sample" / "sales.db"


def test_relative_database_path_is_resolved_from_project_root(tmp_path):
    result = resolve_database_path(
        "data/sample/sales_benchmark_v2.db",
        root_dir=tmp_path,
    )
    expected = (tmp_path / "data" / "sample" / "sales_benchmark_v2.db").resolve()
    assert result == expected


def test_absolute_database_path_is_preserved(tmp_path):
    requested = (tmp_path / "external" / "sales.db").resolve()
    result = resolve_database_path(str(requested), root_dir=tmp_path / "project")
    assert result == requested
