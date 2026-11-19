"""Issue #13: a co-owner cannot take a shared document by deleting its policy
and creating it again.

Failed on rt-baseline (78c72ab) before the fix.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from askiisc.access.directory import Directory
from askiisc.access.policy import DocPolicy, PolicyStore
from askiisc.access.server import OverrideLog
from askiisc.access.service import create_app as create_access
from askiisc.common.types import DocId
from tests.access_fixtures import ISSUER, world


async def test_issue13_delete_then_recreate_keeps_every_owner(
    tmp_path: Path,
) -> None:
    w = world()
    directory = Directory.from_records(
        [
            {"user_id": "a_cds", "department": "cds", "role": "admin"},
            {"user_id": "a_ece", "department": "ece", "role": "admin"},
            {"user_id": "a_csa", "department": "csa", "role": "admin"},
        ]
    )
    shared = DocPolicy(
        doc_id=DocId("shared"),
        owner="cds",
        co_owners=frozenset({"ece"}),
        groups=frozenset({"students"}),
    )
    store = PolicyStore([shared])
    log = OverrideLog(tmp_path / "overrides.jsonl")
    app = create_access(
        directory, store, ISSUER.verifier(w.clock), w.clock, log.record
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://access"
    )

    def auth(user: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {w.token(user)}"}

    sole_ece = {"owner": "ece", "groups": ["students"]}
    url = "/v1/policy/shared"
    control = await client.put(url, json=sole_ece, headers=auth("a_ece"))
    assert control.status_code == 403  # noqa: PLR2004
    revoked = await client.delete(url, headers=auth("a_ece"))
    assert revoked.status_code == 200  # noqa: PLR2004 - a co-owner revokes
    seized = await client.put(url, json=sole_ece, headers=auth("a_ece"))
    assert seized.status_code == 403  # noqa: PLR2004
    # The id stays taken: free to its owners, not to anyone else.
    assert (await client.get(url, headers=auth("a_cds"))).status_code == 404  # noqa: PLR2004
    assert (await client.get(url, headers=auth("a_csa"))).status_code == 403  # noqa: PLR2004
    sole_cds = {"owner": "cds", "groups": ["students"]}
    taken = await client.put(url, json=sole_cds, headers=auth("a_cds"))
    assert taken.status_code == 403  # noqa: PLR2004 - symmetric
    both = {**sole_cds, "co_owners": ["ece"]}
    restored = await client.put(url, json=both, headers=auth("a_cds"))
    assert restored.status_code == 200  # noqa: PLR2004
    await client.delete(url, headers=auth("a_cds"))
    replayed = PolicyStore([shared])
    log.replay(replayed)  # the owners survive a restart of the service
    assert replayed.former_owners(DocId("shared")) == frozenset({"cds", "ece"})
    replayed.delete(DocId("never"))  # no policy: nothing to remember
    assert replayed.former_owners(DocId("never")) is None
    new = await client.put(  # a document that never had a policy
        "/v1/policy/fresh", json=sole_ece, headers=auth("a_ece")
    )
    assert new.status_code == 200  # noqa: PLR2004
