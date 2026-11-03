"""A TEI-compatible embedding server for CPU (TEI itself needs Docker).

POST /embed {"inputs": [...]} returns one unit-length vector per input.
POST /rerank {"query": ..., "texts": [...]} returns one cross-encoder score
per text (when started with --reranker).
It runs in the serve venv (sentence-transformers and torch), started as:

    PYTHONPATH=src $SERVE_VENV/bin/python -m askiisc.retrieval.embed_server \
        --model BAAI/bge-m3 --reranker BAAI/bge-reranker-v2-m3 --port 8003

The project venv never imports sentence-transformers; tests inject `encode`.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from askiisc.common.requestid import log_requests
from askiisc.common.targets import EMBED_MODEL

LOG = logging.getLogger("askiisc.embed.requests")

Encode = Callable[[Sequence[str]], list[list[float]]]
Score = Callable[[str, Sequence[str]], list[float]]
MAX_INPUTS = 256


class EmbedRequest(BaseModel):
    """POST /embed body (the TEI shape)."""

    model_config = ConfigDict(extra="ignore")
    inputs: list[str] = Field(min_length=1, max_length=MAX_INPUTS)


class RerankRequest(BaseModel):
    """POST /rerank body."""

    model_config = ConfigDict(extra="ignore")
    query: str = Field(min_length=1)
    texts: list[str] = Field(min_length=1, max_length=MAX_INPUTS)


def create_app(encode: Encode, score: Score | None = None) -> FastAPI:
    """Builds the server around an encode function.

    Args:
        encode: Maps texts to normalised vectors.
        score: Scores (question, passages); None: no /rerank endpoint.

    Returns:
        A FastAPI application.
    """
    app = FastAPI(title="ask-iisc-embed")
    log_requests(app, "embed", LOG)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/embed")
    async def embed(body: EmbedRequest) -> list[list[float]]:
        try:
            return encode(body.inputs)
        except (RuntimeError, ValueError) as err:
            raise HTTPException(500, detail="encode failed") from err

    @app.post("/rerank")
    async def rerank(body: RerankRequest) -> list[float]:
        if score is None:
            raise HTTPException(404, detail="no reranker loaded")
        try:
            return score(body.query, body.texts)
        except (RuntimeError, ValueError) as err:
            raise HTTPException(500, detail="rerank failed") from err

    return app


def sentence_transformer_encoder(model: str, device: str) -> Encode:
    """Loads a sentence-transformers model (serve venv only)."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]  # noqa: PLC0415, I001

    kwargs = {"torch_dtype": "float16"} if device.startswith("cuda") else {}
    st_model = SentenceTransformer(model, device=device, model_kwargs=kwargs)

    def encode(texts: Sequence[str]) -> list[list[float]]:
        vectors = st_model.encode(
            list(texts),
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [[float(x) for x in v] for v in vectors]

    return encode


def cross_encoder_scorer(model: str, device: str) -> Score:
    """Loads a sentence-transformers cross-encoder (serve venv only)."""
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    kwargs = {"torch_dtype": "float16"} if device.startswith("cuda") else {}
    encoder = CrossEncoder(
        model, device=device, max_length=512, model_kwargs=kwargs
    )

    def score(query: str, texts: Sequence[str]) -> list[float]:
        pairs = [(query, t) for t in texts]
        values = encoder.predict(pairs, batch_size=32, show_progress_bar=False)
        return [float(v) for v in values]

    return score


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    """Runs the server (bound to 127.0.0.1)."""
    import uvicorn  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=EMBED_MODEL)
    parser.add_argument("--reranker", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    score = (
        cross_encoder_scorer(args.reranker, args.device)
        if args.reranker
        else None
    )
    app = create_app(
        sentence_transformer_encoder(args.model, args.device), score
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
