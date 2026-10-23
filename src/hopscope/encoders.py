"""Embedding back-ends. Default is a local ONNX model (no API key)."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .text import tokenize


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class HashingEncoder:
    """Deterministic, dependency-free fallback (feature-hashed unigrams+bigrams).

    Used by the test-suite and CI; not meant to compete with neural embeddings.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _h(self, tok: str) -> int:
        return int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = self._h(f)
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _l2(out)


class FastEmbedEncoder:
    """Local ONNX sentence embeddings via fastembed (default: BAAI/bge-small-en-v1.5)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model)
        self.batch_size = batch_size
        self.name = model

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        it = (self.model.query_embed(texts) if query
              else self.model.embed(texts, batch_size=self.batch_size))
        return _l2(np.asarray(list(it), dtype=np.float32))


def get_encoder(name: str) -> Encoder:
    if name.startswith("hashing"):
        return HashingEncoder()
    return FastEmbedEncoder(name)
