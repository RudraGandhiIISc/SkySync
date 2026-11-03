"""Model comparison page: six models, measured or pending, admins only."""

import json
from pathlib import Path

from askiisc.web.models import comparison, labels, meets, record, target_for
from tests.web_fixtures import build_web

ACTIVE = "qwen3-4b-instruct-2507"


def test_all_six_models_pending_without_results(tmp_path: Path) -> None:
    rows = comparison(tmp_path / "models", ACTIVE)
    assert len(rows) == 6  # noqa: PLR2004
    assert all(r["status"] == "pending" for r in rows)
    assert [r["key"] for r in rows if r["active"]] == [ACTIVE]
    assert comparison(None, "")[0]["metrics"]["accuracy"]["value"] is None
    assert set(labels()) == set(rows[0]["metrics"])


def test_measured_values_and_targets(tmp_path: Path) -> None:
    models = tmp_path / "models"
    path = record(
        models / ACTIVE,
        {
            "accuracy": 0.704,
            "citation_correctness": 0.962,
            "p95_latency_s": 3.03,
            "tokens_per_s": 5195.0,
            "cold_start_s": 25.6,
            "gpu_hours_per_1k": 0.12,
            "unknown_field": 1.0,
        },
        "baseline-v1",
    )
    data = json.loads(path.read_text())
    assert "unknown_field" not in data and data["source"] == "baseline-v1"
    row = next(r for r in comparison(models, ACTIVE) if r["key"] == ACTIVE)
    assert row["status"] == "measured" and row["source"] == "baseline-v1"
    m = row["metrics"]
    assert m["accuracy"] == {"value": 0.704, "target": 0.80, "met": False}
    assert m["citation_correctness"]["met"] is True
    assert m["p95_latency_s"]["met"] is True
    assert m["tokens_per_s"]["target"] == 600.0 and m["tokens_per_s"]["met"]
    assert m["cold_start_s"]["target"] == 90.0
    (models / "qwen3-8b").mkdir()
    (models / "qwen3-8b" / "metrics.json").write_text("not json")
    eight = next(
        r for r in comparison(models, ACTIVE) if r["key"] == "qwen3-8b"
    )
    assert eight["status"] == "pending"


def test_targets_depend_on_size() -> None:
    assert target_for("tokens_per_s", "8B") == 250.0
    assert target_for("tokens_per_s", "8B-AWQ") == 450.0
    assert target_for("cold_start_s", "8B") == 150.0
    assert meets("gpu_hours_per_1k", 0.5, 0.3) is False
    assert meets("accuracy", 0.9, None) is None


async def test_models_api_and_page_for_admins_only(tmp_path: Path) -> None:
    web = build_web(tmp_path)
    web.deps.models_dir = tmp_path / "models"
    web.deps.active_model = ACTIVE
    web.deps.serve_job = "1461"
    student = web.browser()
    web.account("u_cds_stu")
    await web.login("u_cds_stu", student)
    assert (await student.get("/api/admin/models")).status_code == 403
    admin = web.browser()
    web.account("u_admin")
    await web.login("u_admin", admin)
    body = (await admin.get("/api/admin/models")).json()
    assert body["active"] == ACTIVE and len(body["models"]) == 6  # noqa: PLR2004
    assert body["switch"].startswith("scancel 1461 && sbatch")
    assert "ASKIISC_MODEL=<key>" in body["switch"]
    assert "Model comparison" in (await admin.get("/admin/models")).text
    web.deps.serve_job = ""
    body = (await admin.get("/api/admin/models")).json()
    assert body["switch"].startswith("scancel <job>")
