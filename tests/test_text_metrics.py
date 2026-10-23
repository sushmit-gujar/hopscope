from hopscope.metrics import attribution, evidence_recall, full_support, mrr
from hopscope.text import exact_match, extract_entities, f1_score, normalize_entity


def test_normalize_entity_strips_disambiguation_and_articles():
    assert normalize_entity("Ed Wood (film)") == "ed wood"
    assert normalize_entity("The Beatles") == "beatles"


def test_extract_entities_includes_title_and_spans():
    ents = extract_entities("Scott Derrickson is an American director of Doctor Strange.",
                            "Scott Derrickson")
    assert ents[0] == "scott derrickson"
    assert "doctor strange" in ents


def test_answer_metrics_follow_hotpotqa_normalisation():
    assert exact_match("The Beatles", "beatles") == 1.0
    assert f1_score("New York City", "New York") == 0.8
    assert f1_score("", "x") == 0.0


def test_retrieval_metrics():
    assert evidence_recall([1, 2, 3], [3, 9], k=3) == 0.5
    assert full_support([1, 9, 3], [3, 9], k=3) == 1.0
    assert full_support([1, 9, 3], [3, 9], k=2) == 0.0
    assert mrr([5, 6, 9], [9]) == 1 / 3
    assert attribution([9], [3, 9]) == 1.0
    assert attribution([], [3]) == 0.0
