import httpx

from askiisc.common.servicekey import bearer
from askiisc.retrieval.embed_server import create_app
from askiisc.retrieval.embedder import HttpEmbedder


def _encode(texts: object) -> list[list[float]]:
    out = []
    for text in texts:  # type: ignore[attr-defined]
        if text == "boom":
            raise RuntimeError("model failed")
        out.append([1.0, 0.0, 0.0])
    return out


async def test_embed_server_speaks_tei_contract() -> None:
    app = create_app(_encode, key="k")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://e", headers=bearer("k")
    ) as c:
        assert (await c.get("/health")).json() == {"status": "ok"}
        embedder = HttpEmbedder(c, "http://e", dim=3, api_key="k")
        assert await embedder.embed(["a", "b"]) == [(1.0, 0.0, 0.0)] * 2
        assert (
            await c.post("/embed", json={"inputs": ["boom"]})
        ).status_code == 500
        assert (await c.post("/embed", json={"inputs": []})).status_code == 422
