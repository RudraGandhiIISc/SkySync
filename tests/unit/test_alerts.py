"""Grafana alert provisioning (the alerts live in Grafana)."""

import json
from pathlib import Path
from typing import Any

import yaml
from deploy.serve import (
    ALERT_RULES,
    alerting_config,
    alerting_provisioning,
)

RULE_UIDS = {
    "askiisc-latency-p95",
    "askiisc-error-rate",
    "askiisc-locust-leaks",
    "askiisc-worker-failure",
    "askiisc-access-p95",
    "askiisc-target-down",
}


def test_rule_file_has_the_six_alerts() -> None:
    data = yaml.safe_load(ALERT_RULES.read_text())
    rules = data["groups"][0]["rules"]
    assert {r["uid"] for r in rules} == RULE_UIDS
    for rule in rules:
        refs = {d["refId"]: d for d in rule["data"]}
        assert rule["condition"] == "C" and set(refs) == {"A", "C"}
        assert refs["A"]["datasourceUid"] == "askiisc-prom"
        assert refs["A"]["model"]["expr"]
        assert refs["C"]["model"]["type"] == "threshold"
        assert rule["labels"]["requirement"]
    by_uid = {r["uid"]: r for r in rules}
    assert by_uid["askiisc-latency-p95"]["for"] == "5m"
    threshold = by_uid["askiisc-latency-p95"]["data"][1]["model"]
    assert threshold["conditions"][0]["evaluator"]["params"] == [3.5]


def test_notifications_are_off_by_default(tmp_path: Path) -> None:
    config_file = tmp_path / "private" / "alerting.json"
    config = alerting_config(config_file)
    assert config["enabled"] is False
    assert oct(config_file.stat().st_mode)[-3:] == "600"
    root = tmp_path / "prov"
    assert alerting_provisioning(root, config) == {}
    assert (root / "alerting" / "rules.yaml").exists()
    assert not (root / "alerting" / "contact-points.yaml").exists()


def test_email_and_webhook_contact_points(tmp_path: Path) -> None:
    root = tmp_path / "prov"
    config: dict[str, Any] = {
        "enabled": True,
        "email": {
            "to": ["a@x.example", "b@x.example"],
            "from": "askiisc@x.example",
            "smtp_host": "smtp.x.example:587",
            "smtp_user": "u",
            "smtp_password": "p",
        },
        "webhook": {"url": "https://hooks.example/abc"},
    }
    env = alerting_provisioning(root, config)
    assert env["GF_SMTP_ENABLED"] == "true"
    assert env["GF_SMTP_HOST"] == "smtp.x.example:587"
    points = json.loads((root / "alerting" / "contact-points.yaml").read_text())
    receivers = points["contactPoints"][0]["receivers"]
    assert [r["type"] for r in receivers] == ["email", "webhook"]
    assert receivers[0]["settings"]["addresses"] == "a@x.example;b@x.example"
    assert points["policies"][0]["receiver"] == "askiisc"
    # Switching off again removes the contact point.
    assert alerting_provisioning(root, {"enabled": False}) == {}
    assert not (root / "alerting" / "contact-points.yaml").exists()
    # Enabled without a usable receiver: nothing to notify.
    assert (
        alerting_provisioning(root, {"enabled": True, "email": {"to": []}})
        == {}
    )
    assert not (root / "alerting" / "contact-points.yaml").exists()
