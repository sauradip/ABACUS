"""
Fuzzy counting reward for GRPO.

Inspired by CrowdVLM-R1 (arXiv:2504.03724):
- Binary 0/1 reward underperforms SFT
- Fuzzy continuous reward (closer = higher) beats GPT-4o

Design:
  format_reward: 0.0 or 1.0 — did the model output a parseable integer?
  accuracy_reward: continuous in [0, 1] — how close to GT?

  final_reward = 0.1 * format_reward + 0.9 * accuracy_reward

  When format_reward = 0 (unparseable), accuracy_reward = 0 too,
  so final_reward = 0.0 for garbage outputs.

The 0.1 format weight gives a small bonus for outputting ANY number
(even if wrong), preventing the model from collapsing to non-numeric outputs.
The 0.9 accuracy weight dominates — the model must get the count right.
"""

import re


def _extract_text(completion_item):
    if isinstance(completion_item, str):
        return completion_item
    if isinstance(completion_item, list) and completion_item:
        first = completion_item[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
    return str(completion_item)


def counting_reward(model_response, ground_truth, **kwargs):
    """
    Args:
        model_response (str): Model's generated text.
            Expected: a single integer, e.g., "47"
            May also contain: "47\n", "The count is 47", "47 objects", etc.
        ground_truth (str): GT count as string from JSONL "solution" field.
            Always a clean integer string, e.g., "47"

    Returns:
        float: reward in [0.0, 1.0]
    """
    # Parse ground truth.
    # VLM-R1 wraps the JSONL solution in "<answer> N </answer>" before passing
    # it to the reward function, so we must extract the integer from that wrapper.
    try:
        gt_text = ground_truth.strip()
        # Strip <answer>...</answer> wrapper if present
        m = re.search(r'<answer>\s*(.*?)\s*</answer>', gt_text, re.DOTALL)
        if m:
            gt_text = m.group(1).strip()
        gt_count = int(gt_text)
    except (ValueError, AttributeError):
        return 0.0  # Should never happen with well-formed data

    # Parse model response — extract the first integer
    try:
        response = model_response.strip()

        # Strategy 1: Try to parse the entire response as a number
        # This handles clean outputs like "47"
        try:
            pred_count = int(response)
            format_ok = True
        except ValueError:
            # Strategy 2: Extract first integer from text
            # This handles "The answer is 47" or "47 objects"
            match = re.search(r'\b(\d+)\b', response)
            if match:
                pred_count = int(match.group(1))
                format_ok = True
            else:
                format_ok = False
                pred_count = None
    except (AttributeError, TypeError):
        format_ok = False
        pred_count = None

    # Format reward
    format_reward = 1.0 if format_ok else 0.0

    if not format_ok:
        return 0.0

    # Accuracy reward: fuzzy continuous score
    # CrowdVLM-R1 style: reward decreases with relative error
    if gt_count == 0:
        # Edge case: GT is 0
        accuracy_reward = 1.0 if pred_count == 0 else max(0.0, 1.0 - abs(pred_count) / 10.0)
    else:
        relative_error = abs(pred_count - gt_count) / gt_count
        # Clamp: if prediction is >2x off, reward is 0
        accuracy_reward = max(0.0, 1.0 - relative_error)

    # Combined reward
    reward = 0.1 * format_reward + 0.9 * accuracy_reward

    return reward


def counting_reward_func(completions, **kwargs):
    """Open-R1 reward adapter — same interface as count_consistency_reward_func."""
    gt_list = kwargs.get("solution", [])
    rewards = []
    for completion_item, gt in zip(completions, gt_list):
        response_text = _extract_text(completion_item)
        rewards.append(counting_reward(response_text, gt))
    return rewards


if __name__ == "__main__":
    # Standalone tests
    assert counting_reward("47", "47") == 1.0, "perfect"
    assert counting_reward("50", "47") > 0.9, "close"
    assert counting_reward("94", "47") <= 0.1, "2x off"
    assert counting_reward("0", "47") == 0.1, "completely wrong — gets format bonus only"
    assert counting_reward("blah", "47") == 0.0, "unparseable"
    assert counting_reward("The answer is 47", "47") == 1.0, "extracted"
    assert counting_reward("I count about 45 objects", "47") > 0.9, "close, extracted"
    assert counting_reward("0", "0") == 1.0, "edge case"
    print("All reward function tests passed.")
