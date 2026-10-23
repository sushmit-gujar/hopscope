import json

import httpx

from hopscope.data import Passage
from hopscope.readers import LLMReader

P = [Passage(7, "Jane Doe", "Jane Doe was born in Springfield."),
     Passage(9, "Acme", "Acme Rocket was designed by Jane Doe.")]


def _client(payload: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Question:" in json.loads(request.content)["messages"][0]["content"]
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_reader_parses_answer_and_citations(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    content = json.dumps({"answer": "Springfield", "citations": [1, 2, 5]})
    r = LLMReader("openai", "x", client=_client({"choices": [{"message": {"content": content}}]}))
    a = r.answer("Where was the designer of Acme Rocket born?", P)
    assert a.text == "Springfield"
    assert a.source_pids == [7, 9]  # out-of-range citation 5 dropped


def test_anthropic_reader(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    text = 'Sure: {"answer": "Springfield", "citations": [1]}'
    r = LLMReader("anthropic", "x", client=_client({"content": [{"text": text}]}))
    assert r.answer("q?", P).source_pids == [7]
