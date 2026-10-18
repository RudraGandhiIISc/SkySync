"""A TEI-compatible embedding server for CPU (TEI itself needs Docker).

POST /embed {"inputs": [...]} returns one unit-length vector per input.
It runs in the serve venv (sentence-transformers and torch), started as:

    PYTHONPATH=src $SERVE_VENV/bin/python -m askiisc.retrieval.embed_server \
        --model BAAI/bge-base-en-v1.5 --port 8003

The project venv never imports sentence-transformers; tests inject `encode`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

Encode = Callable[[Sequence[str]], list[list[float]]]
MAX_INPUTS = 256


class EmbedRequest(BaseModel):
    """POST /embed body (the TEI shape)."""

    model_config = ConfigDict(extra="ignore")
    inputs: list[str] = Field(min_length=1, max_length=MAX_INPUTS)


def create_app(encode: Encode) -> FastAPI:
    """Builds the server around an encode function.

    Args:
        encode: Maps texts to normalised vectors.

    Returns:
        A FastAPI application.
    """
    app = FastAPI(title="ask-iisc-embed")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/embed")
    async def embed(body: EmbedRequest) -> list[list[float]]:
        try:
            return encode(body.inputs)
        except (RuntimeError, ValueError) as err:
            raise HTTPException(500, detail="encode failed") from err

    return app


def sentence_transformer_encoder(model: str, device: str) -> Encode:
    """Loads a sentence-transformers model (serve venv only)."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]  # noqa: PLC0415, I001

    st_model = SentenceTransformer(model, device=device)

    def encode(texts: Sequence[str]) -> list[list[float]]:
        vectors = st_model.encode(
            list(texts),
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [[float(x) for x in v] for v in vectors]

    return encode


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    """Runs the server (bound to 127.0.0.1)."""
    import uvicorn  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args(argv)
    app = create_app(sentence_transformer_encoder(args.model, args.device))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
