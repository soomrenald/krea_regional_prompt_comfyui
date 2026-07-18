from krea_regional_prompt_comfyui.k2_region_core.regional_prompting import (
    krea_prompt_token_count,
)


def tokenized(*token_ids, key="qwen3vl_4b"):
    return {key: [[(token_id, 1.0, 0) for token_id in token_ids]]}


def test_counts_prompt_inside_complete_krea_chat_wrapper():
    tokens = tokenized(
        151644,
        8948,
        198,
        100,
        151645,
        198,
        151644,
        872,
        198,
        41,
        42,
        151645,
        198,
        151644,
        77091,
        198,
    )
    assert krea_prompt_token_count(tokens) == 2


def test_counts_prompt_when_wrapper_removed_user_im_end():
    tokens = tokenized(
        151644,
        8948,
        198,
        100,
        151645,
        198,
        151644,
        872,
        198,
        41,
        42,
        198,
        151644,
        77091,
        198,
    )
    assert krea_prompt_token_count(tokens) == 2


def test_counts_already_stripped_prompt_tokens():
    assert krea_prompt_token_count(tokenized(41, 42)) == 2


def test_prefers_qwen_wrapper_when_loader_returns_multiple_groups():
    tokens = {
        "auxiliary": [[(1, 1.0, 0), (2, 1.0, 0)]],
        **tokenized(151644, 1, 151645, 151644, 872, 198, 41, 151645),
    }
    assert krea_prompt_token_count(tokens) == 1
