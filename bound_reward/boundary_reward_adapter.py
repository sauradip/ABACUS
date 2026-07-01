"""
Adapter that wraps CompositeReward + StructuredOutputParser into the
GRPO reward function signature expected by run_grpo_v3.py:

    reward_func(prompts, completions, gt_count, **kwargs) -> List[float]

Register in REWARD_REGISTRY as "boundary_decomposed_reward".
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make bound_reward/ importable regardless of CWD
_BOUND_REWARD_DIR = Path(__file__).resolve().parent
if str(_BOUND_REWARD_DIR) not in sys.path:
    sys.path.insert(0, str(_BOUND_REWARD_DIR))

from boundary_reward import CompositeReward, RewardWeights  # noqa: E402
from prompt_template import StructuredOutputParser           # noqa: E402


_parser  = StructuredOutputParser()
_reward  = CompositeReward(RewardWeights(
    R3_overcount   = 0.40,
    R_conservation = 0.25,
    R_sum          = 0.20,
    R_outcome      = 0.15,
))

_QUADRANT_KEYS   = ("top_left", "top_right", "bottom_left", "bottom_right")
_BOUNDARY_KEYS   = ("boundary_claimed", "boundary_discarded")


def _format_score(text: str) -> float:
    """
    Partial credit for producing boundary-format text, even if not parseable.
    Scores 0..7 based on presence of expected structural keywords.
    This creates reward variance during the cold-start phase before the model
    learns to emit valid JSON, preventing reward_std=0 / zero GRPO loss.
    """
    score = 0.0
    if "{" in text and "}" in text:
        score += 1.0
    for key in _QUADRANT_KEYS:
        if key in text:
            score += 1.0
    for key in _BOUNDARY_KEYS:
        if key in text:
            score += 0.5
    return score  # 0..6


def boundary_decomposed_reward(
    prompts,
    completions,
    gt_count,
    **kwargs,
) -> list[float]:
    """
    GRPO reward function for boundary-aware decomposed counting.

    Each completion is a structured JSON output from CountingPromptBuilder:
      {reasoning: {...}, counts: {top_left: {...}, ...}, total: N}

    Returns reward in roughly [-10, 0]; 0 = perfect boundary assignment.

    On parse failure: partial credit from _format_score() is added to -10,
    giving range [-10, -4]. This breaks reward_std=0 during cold-start so
    GRPO can learn the output format before full parse succeeds.
    """
    rewards = []
    for completion, gt in zip(completions, gt_count):
        # completions arrive as list[dict] or raw string depending on TRL version
        if isinstance(completion, list):
            text = completion[0].get("content", "") if completion else ""
        else:
            text = str(completion)

        parsed    = _parser.parse(text)
        breakdown = _reward.compute(parsed, gt_total=int(gt))

        if not parsed.parse_success:
            # Partial format credit lifts score from -10 toward -4 based on
            # how many expected structural keywords appear in the output.
            score = -10.0 + _format_score(text)
        else:
            score = max(breakdown.R_composite, -10.0)

        rewards.append(score)

    return rewards
