"""Analytical scoring cases: no model judge or post-result equivalence additions."""
import os

import pytest

from bench.cached_decode import metrics, normalize, semantic_forms, trace


def row(gold, value, kind="actor"):
    return {"gold": gold, "semantic_key": value, "kind": kind}


def test_exact_prefix_and_complete_value_are_separate():
    r = row("Mary Jane", "Mary Jane")
    assert metrics("Mary", r) == {"exact_literal": False, "prefix_agreement": True,
                                 "strict_partial_prefix": True, "semantic_value": False}
    assert metrics("Mary Jane", r)["exact_literal"]
    assert metrics("mary jane", r)["semantic_value"]
    assert not metrics("Mary Jane.", r)["semantic_value"]
    assert not metrics(" Mary Jane", r)["semantic_value"]
    assert not metrics("", r)["prefix_agreement"]
    assert not metrics("Mary Jane reported it", r)["semantic_value"]


def test_only_case_articles_and_frozen_synonyms_normalize():
    assert normalize("The image") == normalize("an image") == "image"
    assert normalize("the the image") == "the image"  # At most one leading article.
    r = row("recovered", "restored", "action")
    assert semantic_forms(r) == {"restored", "recovered"}
    assert metrics("Restored", r)["semantic_value"]
    assert metrics("Recovered", r)["semantic_value"]
    assert not metrics("Rest", r)["semantic_value"]
    assert not metrics("restored the image", r)["semantic_value"]
    assert not metrics("restored.", r)["semantic_value"]
    assert not metrics("retrieved", r)["semantic_value"]


def test_multiword_alias_and_optional_gold_articles():
    assert metrics("Rolled out", row("deployed", "deployed", "action"))["semantic_value"]
    assert not metrics("Rolled", row("deployed", "deployed", "action"))["semantic_value"]
    r = row("the image", "image", "object")
    assert metrics("image", r)["semantic_value"]
    assert not metrics("image", r)["exact_literal"]
    assert metrics("the", r)["prefix_agreement"]
    assert not metrics("the", r)["semantic_value"]


@pytest.mark.engine
@pytest.mark.skipif(os.getenv("ANYJEV_CACHED_ENGINE") != "1", reason="optional random-model cache parity")
def test_native_cache_matches_greedy_and_really_consumes_single_tokens():
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig, Qwen2Config

    torch.manual_seed(73)
    model = AutoModelForCausalLM.from_config(Qwen2Config(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        eos_token_id=63, pad_token_id=0)).eval()
    model.requires_grad_(False)
    class Decoder:
        def decode(self, ids, skip_special_tokens=True):
            return ":".join(str(i) for i in ids if not skip_special_tokens or i != 63)
    shapes = []
    handle = model.register_forward_pre_hook(lambda m, args, kwargs: shapes.append(kwargs["input_ids"].shape[1]),
                                             with_kwargs=True)
    ids = [4, 7, 12, 17]
    steps = trace(model, Decoder(), ids)
    handle.remove()
    assert shapes == [len(ids)] + [1] * (len(steps) - 1)
    with torch.inference_mode():
        generated = model.generate(torch.tensor([ids]), generation_config=GenerationConfig(
            max_new_tokens=6, do_sample=False, eos_token_id=63, pad_token_id=0))
    assert [s["token_id"] for s in steps] == generated[0, len(ids):].tolist()
    assert all(s["cache_length"] == len(ids) + s["step"] - 1 for s in steps)
