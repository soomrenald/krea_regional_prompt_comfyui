from krea_regional_prompt_comfyui.k2_region_core.regional_prompting import (
    PromptEmphasis,
    compile_regional_prompt_plan,
    krea_prompt_token_count,
)
from krea_regional_prompt_comfyui.k2_region_core.regions import (
    PixelBox,
    RegionDefinition,
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


def test_prompt_emphasis_excludes_tokenized_leading_space_from_start():
    region = RegionDefinition(
        "subject", "Subject", PixelBox(0, 0, 32, 64), "woman in foreground"
    )
    plan = compile_regional_prompt_plan(
        64,
        64,
        "gallery",
        (region,),
        emphases=(PromptEmphasis("subject", "foreground", 0.8),),
    )

    def qwen_like_prefix_count(prefix):
        words = len(prefix.rstrip().split())
        return words + int(bool(prefix) and prefix[-1].isspace())

    bound = plan.bind_tokens(qwen_like_prefix_count)

    assert bound.emphases[0].end - bound.emphases[0].start == 1
