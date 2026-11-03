"""The throughput sweep's prompts come from either results format."""

from eval.llm_bench import _retrieved


def test_retrieved_chunks_are_read_from_both_formats() -> None:
    heldout = {"answer": {"retrieved": [["d1@v2#0", "d1"]]}}
    versioned = {"answer": "text", "retrieved": ["lib-x@v1#3", "d2@v4#0"]}
    assert _retrieved(heldout) == [("d1@v2#0", "d1")]
    assert _retrieved(versioned) == [("lib-x@v1#3", "lib-x"), ("d2@v4#0", "d2")]
    assert _retrieved({"answer": None}) == []
