"""Titles of uploaded documents survive a restart of the serving job."""

import json
from pathlib import Path

from deploy.serve import UPLOADS_FILE, load_uploads, save_upload


def test_uploaded_titles_are_kept_on_disk(tmp_path: Path) -> None:
    assert load_uploads(tmp_path) == {}  # no file yet
    info = {"title": "rules.pdf", "url": "", "kind": "pdf", "site": "uploaded"}
    save_upload(tmp_path, "csa-lab-rules", info)
    save_upload(tmp_path, "cds-notes", {**info, "title": "notes.txt"})
    again = load_uploads(tmp_path)  # what the next serving job reads
    assert again["csa-lab-rules"] == info
    assert again["cds-notes"]["title"] == "notes.txt"
    assert oct((tmp_path / UPLOADS_FILE).stat().st_mode)[-3:] == "600"
    (tmp_path / UPLOADS_FILE).write_text("{broken")
    assert load_uploads(tmp_path) == {}  # an unreadable file means none
    (tmp_path / UPLOADS_FILE).write_text(json.dumps({"x": "not a dict"}))
    assert load_uploads(tmp_path) == {}
