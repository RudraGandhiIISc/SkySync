"""Serves the web app on fakes with demo data, for UI screenshots.

Run inside Slurm (slurm/ui_preview.sh does it and takes the screenshots):

    python -m tools.ui_preview PORT

Everything is in memory or in a temporary folder: the fake gateway stack
of the tests, a demo users tree, a few chats and some request-log rows.
No real users, corpus or model. Logins all use tests.web_fixtures.PASSWORD.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import uvicorn
from tests.access_fixtures import MATRIX
from tests.web_fixtures import HDR, build_web

from askiisc.gateway.app import RequestEvent
from askiisc.web.documents import find_documents
from askiisc.web.people import PeopleTree

TREE: dict[str, list[dict[str, Any]]] = {
    "institute/admins.json": [{"user_id": "u_admin"}],
    "cds/admins.json": [{"user_id": "u_cds_admin"}],
    "cds/students.json": [
        {"user_id": "u_cds_stu", "program": "mtech-cds", "joined": 2025},
        {"user_id": "u_phd", "program": "phd", "joined": 2023},
        {"user_id": "u_banned", "program": "mtech-cds", "joined": 2024},
        {
            "user_id": "real.person@iisc.example",
            "program": "mtech-research",
            "joined": 2024,
        },
        {
            "user_id": "ananya.rao@iisc.example",
            "program": "phd",
            "joined": 2022,
        },
        {
            "user_id": "vikram.s@iisc.example",
            "program": "mtech-cds",
            "joined": 2025,
        },
    ],
    "cds/faculty.json": [
        {"user_id": "u_cds_fac", "kind": "faculty", "joined": 2015},
        {
            "user_id": "guest.fac@iisc.example",
            "kind": "visiting",
            "joined": 2026,
        },
    ],
    "cds/staff.json": [
        {"user_id": "u_staff", "kind": "research", "joined": 2025}
    ],
    "csa/admins.json": [{"user_id": "csa.admin@iisc.example"}],
    "csa/students.json": [
        {"user_id": "u_csa_stu", "program": "mtech-cse", "joined": 2025},
        {
            "user_id": "u_hostel",
            "program": "phd",
            "joined": 2021,
            "groups": ["hostel_committee"],
        },
    ],
    "csa/faculty.json": [{"user_id": "u_csa_fac", "kind": "faculty"}],
}

CHATS = [
    (
        "M.Tech credit requirements",
        "What are the minimum credits needed for an M.Tech degree at CDS?",
        "An M.Tech student at CDS needs **at least 64 credits** to graduate "
        "[chunk:d01@v1#0]:\n\n- **Core courses:** 20 credits, taken in the "
        "first two terms.\n- **Electives:** 20 credits, at least 12 from "
        "the department [chunk:d01@v1#0].\n- **Project:** 24 credits over the "
        "second year [chunk:d02@v1#0].\n\nA CGPA of 5.0 or more is needed in "
        "each term to stay in good standing [chunk:d02@v1#0].",
        [
            ("d01", "CDS M.Tech curriculum 2025", 3, 4, False),
            ("d02", "Scheme of instruction: M.Tech programmes", 2, 11, True),
        ],
    ),
    (
        "Hostel fee payment",
        "How do I pay the hostel fee and what is the deadline?",
        "Hostel fees are paid each term through the institute's online fee "
        "portal [chunk:d05@v1#0]. The deadline is the **last working day of "
        "the first week** of the term; a late fee applies after that "
        "[chunk:d05@v1#0].",
        [("d05", "Hostel office: fee circular", 1, 1, False)],
    ),
    ("Library hours in the exam week", "", "", []),
    ("PhD comprehensive exam", "", "", []),
]


def _tree(root: Path) -> Path:
    for name, records in TREE.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(records))
    return root


def _seed_chats(web: Any, user: str, now: float) -> None:
    store = web.deps.store
    for n, (title, question, answer, cites) in enumerate(CHATS):
        chat = store.create_chat(user, title, now - n * 86400 * 3)
        if not question:
            continue
        store.add_message(chat.chat_id, "user", question, {}, now - 60)
        citations = [
            {
                "doc_id": doc,
                "title": name,
                "url": f"https://www.iisc.example/{doc}",
                "link": f"https://www.iisc.example/{doc}.pdf#page={page}"
                if page > 1
                else f"https://www.iisc.example/{doc}#:~:text={doc}",
                "kind": "pdf" if page > 1 else "html",
                "site": "www.iisc.example",
                "version": version,
                "page": page,
                "chunk_id": f"{doc}@v1#0",
                "is_stale": stale,
            }
            for doc, name, version, page, stale in cites
        ]
        meta = {
            "citations": citations,
            "abstained": False,
            "cache_hit": False,
            "stale": any(c["is_stale"] for c in citations),
            "model": "Qwen3-4B-Instruct-2507",
        }
        store.add_message(chat.chat_id, "assistant", answer, meta, now - 55)


def _seed_requests(web: Any, now: float) -> None:
    users = [("u_cds_stu", "cds"), ("u_phd", "cds"), ("u_csa_stu", "csa")]
    for i in range(240):
        user, dept = users[i % 3]
        decision = "error" if i % 37 == 0 else "answered"
        web.deps.requests.record(
            RequestEvent(
                request_id=f"r{i:04d}demo",
                time=now - 3500 + i * 14.5,
                user_id=user,
                tenant=dept,
                source="chat" if i % 4 else "temporary",
                synthetic=i % 5 == 0,
                decision=decision,
                status=500 if decision == "error" else 200,
                latency_ms=1400.0 + (i * 37) % 1900,
                ttft_ms=300.0 + (i * 13) % 400,
                prompt_tokens=900,
                completion_tokens=120 + i % 80,
                cache_hit=i % 6 == 0,
                stale=False,
                citations=2,
                documents=("d1@v3",),
                model="Qwen3-4B-Instruct-2507",
                policy_version=1,
                cold_hold=False,
            )
        )
    web.deps.requests.flush()


def _system(now: float) -> dict[str, Any]:
    """A serving job like the live one: our worker on GPU 0, others busy."""
    gib = 1024
    return {
        "model": "qwen3-8b",
        "vllm_version": "0.11.0",
        "workers": [{"worker_id": "vllm-1", "gpu": 0, "state": "ready"}],
        "queue": {"waiting": 0, "running": 3, "kv_cache": 0.38,
                  "tokens_per_s": 142.0},
        "gpus": [
            {"index": 0, "utilization": 100, "memory_used_mib": 20.3 * gib,
             "memory_total_mib": 140.4 * gib, "ours": True},
            {"index": 1, "utilization": 100, "memory_used_mib": 139.4 * gib,
             "memory_total_mib": 140.4 * gib, "ours": False},
            {"index": 2, "utilization": 0, "memory_used_mib": 0.5 * gib,
             "memory_total_mib": 140.4 * gib, "ours": False},
        ],
        "chunks": 574672,
        "documents": 30762,
        "started": now - 120,
        "links": {"grafana": "http://localhost:13000",
                  "prometheus": "http://localhost:19090",
                  "locust": "http://localhost:18089"},
    }  # fmt: skip


async def _main(port: int) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="askiisc-ui-"))
    web = build_web(tmp)
    clock = web.stack.w.clock
    clock.advance(time.time() - clock.now())
    web.deps.system = lambda: _system(clock.now())
    catalogue = {
        d: {
            **web.deps.docs(d),
            "site": "cds.iisc.ac.in" if i % 3 else "www.csa.iisc.ac.in",
            "kind": "pdf" if i % 2 else "html",
            "edition": "2025-26" if i % 4 == 0 else "",
        }
        for i, d in enumerate(MATRIX["documents"])
    }
    web.deps.find_docs = lambda q, n, s: find_documents(catalogue, q, n, s)
    root = _tree(tmp / "users")
    web.deps.directory.load(root)
    web.deps.people = PeopleTree(root, web.deps.directory)
    for user in ("u_cds_stu", "u_admin", "u_cds_admin", "u_phd", "u_csa_stu"):
        web.account(user)
    now = clock.now()
    _seed_chats(web, "u_cds_stu", now)
    _seed_requests(web, now)
    await web.login("u_admin")
    await web.client.post(
        "/api/admin/people",
        json={"user_id": "new.student@iisc.example", "department": "cds",
              "role": "student", "program": "phd", "joined": 2026,
              "groups": []},
        headers=HDR,
    )  # fmt: skip
    app = web.client._transport.app  # type: ignore[attr-defined]  # noqa: SLF001
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    print(f"ui preview on http://127.0.0.1:{port}", flush=True)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(_main(int(sys.argv[1])))
