"""Readers turn retrieved passages into an answer + the passage it came from (attribution).

* ``OnnxExtractiveReader`` (default): DistilBERT-SQuAD, quantised ONNX, CPU, no key.
* ``LexicalReader``: deterministic fallback used in tests/CI.
* ``LLMReader``: optional OpenAI / Anthropic generator selected by env vars.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np

from .data import Passage
from .text import tokenize


@dataclass
class Answer:
    text: str
    source_pids: list[int] = field(default_factory=list)
    score: float = 0.0


class Reader(Protocol):
    name: str

    def answer(self, question: str, passages: list[Passage]) -> Answer: ...


class LexicalReader:
    """Pick the passage with most query overlap and return its first unseen capitalised span."""

    name = "lexical"

    def answer(self, question: str, passages: list[Passage]) -> Answer:
        if not passages:
            return Answer("")
        q = set(tokenize(question))
        best = max(passages, key=lambda p: len(q & set(tokenize(p.full))))
        for m in re.finditer(r"\b[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*", best.text):
            cand = m.group(0)
            if not set(tokenize(cand)) & q:
                return Answer(cand, [best.pid])
        return Answer(best.title, [best.pid])


class OnnxExtractiveReader:
    name = "distilbert-squad-onnx"

    def __init__(self, repo: str = "Xenova/distilbert-base-cased-distilled-squad",
                 max_len: int = 384, max_answer_tokens: int = 30) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len, strategy="only_second")
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.sess = ort.InferenceSession(hf_hub_download(repo, "onnx/model_quantized.onnx"), opts)
        self.max_answer_tokens = max_answer_tokens

    def answer(self, question: str, passages: list[Passage]) -> Answer:
        if not passages:
            return Answer("")
        encs = self.tok.encode_batch([(question, p.full) for p in passages])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        start, end = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})
        best = (-1e9, 0, 0, 0)
        for b, e in enumerate(encs):
            ctx = np.array([s == 1 for s in e.sequence_ids])
            if not ctx.any():
                continue
            s_log = np.where(ctx, start[b], -1e9)
            e_log = np.where(ctx, end[b], -1e9)
            for si in np.argsort(-s_log)[:20]:
                hi = min(len(e_log), si + self.max_answer_tokens)
                if si >= hi:
                    continue
                ei = si + int(np.argmax(e_log[si:hi]))
                sc = float(s_log[si] + e_log[ei])
                if sc > best[0]:
                    best = (sc, b, int(si), ei)
        sc, b, si, ei = best
        if sc <= -1e8:
            return Answer("")
        text = passages[b].full
        a, z = encs[b].offsets[si][0], encs[b].offsets[ei][1]
        return Answer(text[a:z].strip(), [passages[b].pid], sc)


_PROMPT = """Answer the question using ONLY the numbered passages. Reply with JSON:
{{"answer": "<short answer, a few words, or 'unanswerable'>", "citations": [<passage numbers>]}}

{ctx}

Question: {q}"""


class LLMReader:
    """Optional hosted-LLM reader. Set HOPSCOPE_LLM_PROVIDER=openai|anthropic and the
    matching OPENAI_API_KEY / ANTHROPIC_API_KEY. Model via HOPSCOPE_LLM_MODEL."""

    def __init__(self, provider: str | None = None, model: str | None = None,
                 client: httpx.Client | None = None) -> None:
        self.provider = (provider or os.environ.get("HOPSCOPE_LLM_PROVIDER", "openai")).lower()
        default = "gpt-4o-mini" if self.provider == "openai" else "claude-3-5-haiku-latest"
        self.model = model or os.environ.get("HOPSCOPE_LLM_MODEL", default)
        self.client = client or httpx.Client(timeout=60)
        self.name = f"{self.provider}:{self.model}"

    def _call(self, prompt: str) -> str:
        if self.provider == "openai":
            r = self.client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={"model": self.model, "temperature": 0,
                      "response_format": {"type": "json_object"},
                      "messages": [{"role": "user", "content": prompt}]},
            )
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"])
        if self.provider == "anthropic":
            r = self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                         "anthropic-version": "2023-06-01"},
                json={"model": self.model, "max_tokens": 200, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
            )
            r.raise_for_status()
            return str(r.json()["content"][0]["text"])
        raise ValueError(f"unknown provider {self.provider!r}")

    def answer(self, question: str, passages: list[Passage]) -> Answer:
        ctx = "\n".join(f"[{i + 1}] {p.full}" for i, p in enumerate(passages))
        raw = self._call(_PROMPT.format(ctx=ctx, q=question))
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            obj = json.loads(m.group(0) if m else raw)
        except json.JSONDecodeError:
            return Answer(raw.strip()[:100])
        cites = [passages[int(c) - 1].pid for c in obj.get("citations", [])
                 if str(c).isdigit() and 0 < int(c) <= len(passages)]
        return Answer(str(obj.get("answer", "")), cites)


def get_reader(name: str) -> Reader:
    if name == "lexical":
        return LexicalReader()
    if name == "llm":
        return LLMReader()
    return OnnxExtractiveReader()
