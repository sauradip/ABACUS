"""
Scaffold-R1 reward functions for grounded counting GRPO.

Reward components:
  - Count reward: exact or relative count accuracy from <|count|>
  - Localization reward: inverse Chamfer distance on normalized [0, 1000] points
  - Format reward: scaffold tag compliance and numeric count parsing

This module is written to be tolerant of several ground-truth payload shapes so it
can be used both with the scaffold JSONL emitted by fsc_to_scaffold.py and with
Open-R1 style wrappers that pass fields through kwargs.
"""

import json
import math
import re


THOUGHT_TAG = "<|thought|>"
SCAFFOLD_TAG = "<|scaffold|>"
COUNT_TAG = "<|count|>"
ANSWER_TAG = "<|answer|>"


def _extract_text(completion_item):
    if isinstance(completion_item, str):
        return completion_item
    if isinstance(completion_item, list) and completion_item:
        first = completion_item[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
    return str(completion_item)


def _unwrap_answer_block(text):
    if not isinstance(text, str):
        return text
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _parse_count(text):
    if not isinstance(text, str):
        return None

    text = _unwrap_answer_block(text)
    tag_match = re.search(r"<\|count\|>\s*([-+]?\d+)", text)
    if tag_match:
        try:
            return int(tag_match.group(1))
        except ValueError:
            return None

    stripped = text.strip()
    if re.fullmatch(r"[-+]?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            return None

    loose_match = re.search(r"\b(\d+)\b", stripped)
    if loose_match:
        try:
            return int(loose_match.group(1))
        except ValueError:
            return None
    return None


def _parse_points_from_text(text):
    if not isinstance(text, str):
        return []

    text = _unwrap_answer_block(text)
    scaffold_match = re.search(r"<\|(?:scaffold|code)\|>\s*(\[\[.*?\]\])\s*(?:<\|count\|>|$)", text, re.DOTALL)
    candidate = scaffold_match.group(1).strip() if scaffold_match else None

    if candidate is not None:
        try:
            parsed = json.loads(candidate)
            return _normalize_point_list(parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    pair_matches = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    if not pair_matches:
        return []
    return [[int(round(float(x_coord))), int(round(float(y_coord)))] for x_coord, y_coord in pair_matches]


def _normalize_point_list(point_list):
    if not isinstance(point_list, list):
        return []

    normalized = []
    for point in point_list:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            x_coord = int(round(float(point[0])))
            y_coord = int(round(float(point[1])))
        except (TypeError, ValueError):
            continue
        normalized.append([x_coord, y_coord])
    return normalized


def _parse_ground_truth_points(ground_truth_text=None, normalized_points=None):
    if normalized_points is not None:
        return _normalize_point_list(normalized_points)
    if isinstance(ground_truth_text, str):
        return _parse_points_from_text(ground_truth_text)
    return []


def _parse_ground_truth_count(ground_truth_text=None, explicit_count=None, points=None):
    if explicit_count is not None:
        try:
            return int(explicit_count)
        except (TypeError, ValueError):
            pass

    parsed = _parse_count(ground_truth_text)
    if parsed is not None:
        return parsed

    if points is not None:
        return len(points)
    return 0


def _parse_solution_int(solution_value):
    if isinstance(solution_value, (int, float)):
        return int(solution_value)
    if isinstance(solution_value, str):
        parsed = _parse_count(solution_value)
        if parsed is not None:
            return parsed
        stripped = _unwrap_answer_block(solution_value)
        if re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
    return None


def _squared_distance(point_a, point_b):
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return dx * dx + dy * dy


def _mean_min_distance(source_points, target_points):
    if not source_points or not target_points:
        return None

    total = 0.0
    for source_point in source_points:
        total += math.sqrt(min(_squared_distance(source_point, target_point) for target_point in target_points))
    return total / len(source_points)


def chamfer_distance(points_a, points_b):
    if not points_a and not points_b:
        return 0.0
    if not points_a or not points_b:
        return 1000.0

    forward = _mean_min_distance(points_a, points_b)
    backward = _mean_min_distance(points_b, points_a)
    return 0.5 * (forward + backward)


def format_reward(model_response):
    if not isinstance(model_response, str):
        return 0.0

    reward = 0.0
    response = _unwrap_answer_block(model_response)
    if THOUGHT_TAG in response and SCAFFOLD_TAG in response and COUNT_TAG in response:
        reward += 0.2
    if _parse_count(response) is None:
        reward -= 0.5
    return reward


def count_value_reward(model_response, ground_truth_count):
    predicted_count = _parse_count(model_response)
    if predicted_count is None:
        return 0.0

    if predicted_count == ground_truth_count:
        return 1.0

    if ground_truth_count <= 0:
        return max(0.0, 1.0 - abs(predicted_count - ground_truth_count))

    return max(0.0, 1.0 - (abs(predicted_count - ground_truth_count) / ground_truth_count))


def count_reward(completions, solution, **kwargs):
    del kwargs
    rewards = []
    answer_tag_pattern = r"<answer>(\d+)</answer>"

    for completion_item, sol in zip(completions, solution):
        content = _extract_text(completion_item)
        reward = 0.0
        answer_number = 0

        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                answer_number = int(content_answer_match.group(1))
                solution_value = _parse_solution_int(sol)
                if solution_value is not None and solution_value != 0:
                    abs_gap = abs(solution_value - answer_number) / solution_value
                    if abs_gap < 0.5:
                        reward = 1.5 - abs_gap
        except Exception:
            reward = 0.0

        if not isinstance(reward, (int, float)) or reward < 0 or reward > 1_000_000 or math.isinf(reward):
            reward = 0.0

        rewards.append(float(reward))
    return rewards


def count_format_reward(completions, **kwargs):
    del kwargs
    pattern = r"<think>.*?</think>\s*<answer>(\d+)</answer>"
    completion_contents = [_extract_text(completion_item) for completion_item in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def localization_reward(model_response, ground_truth_points):
    predicted_points = _parse_points_from_text(model_response)
    if not predicted_points or not ground_truth_points:
        return 0.0

    distance = chamfer_distance(predicted_points, ground_truth_points)
    return max(0.0, 1.0 - (distance / 1000.0))


def scaffold_grpo_reward(model_response, ground_truth_text=None, ground_truth_count=None, normalized_points_1000=None):
    gt_points = _parse_ground_truth_points(ground_truth_text, normalized_points_1000)
    gt_count = _parse_ground_truth_count(ground_truth_text, ground_truth_count, gt_points)

    reward_count = count_value_reward(model_response, gt_count)
    reward_loc = localization_reward(model_response, gt_points)
    reward_format = format_reward(model_response)
    total_reward = reward_count + reward_loc + reward_format

    return {
        "count_reward": reward_count,
        "localization_reward": reward_loc,
        "format_reward": reward_format,
        "total_reward": total_reward,
    }


def compute_scaffold_reward(completion_text, target_points, target_count):
    reward_parts = scaffold_grpo_reward(
        completion_text,
        ground_truth_count=target_count,
        normalized_points_1000=target_points,
    )
    return reward_parts["total_reward"]


def group_normalize_rewards(rewards, epsilon=1e-6):
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    std_reward = math.sqrt(max(variance, 0.0))
    if std_reward < epsilon:
        return [0.0 for _ in rewards]
    return [(reward - mean_reward) / std_reward for reward in rewards]


def scaffold_grpo_reward_func(completions, **kwargs):
    gt_solutions = kwargs.get("solution", [])
    gt_counts = kwargs.get("ground_truth_count", [])
    gt_points = kwargs.get("normalized_points_1000", [])

    rewards = []
    for index, completion_item in enumerate(completions):
        response_text = _extract_text(completion_item)
        solution_text = gt_solutions[index] if index < len(gt_solutions) else None
        count_value = gt_counts[index] if index < len(gt_counts) else None
        points_value = gt_points[index] if index < len(gt_points) else None
        reward_parts = scaffold_grpo_reward(
            response_text,
            ground_truth_text=solution_text,
            ground_truth_count=count_value,
            normalized_points_1000=points_value,
        )
        rewards.append(reward_parts["total_reward"])
    return rewards


def reward_function(prompts, completions, **kwargs):
    del prompts
    gt_points = kwargs.get("gt_points", kwargs.get("normalized_points_1000", []))
    gt_counts = kwargs.get("gt_count", kwargs.get("ground_truth_count", []))

    rewards = []
    for completion, target_points, target_count in zip(completions, gt_points, gt_counts):
        completion_text = _extract_text(completion)
        rewards.append(compute_scaffold_reward(completion_text, target_points, target_count))
    return rewards


reward_funcs_registry = {
    "scaffold_r1": scaffold_grpo_reward_func,
    "counting_fuzzy": count_reward,
    "counting_format": count_format_reward,
}


if __name__ == "__main__":
    gt = (
        "<|thought|>\nCount left-to-right.\n"
        "<|answer|>\n<|scaffold|> [[10,10],[30,30],[50,50]]\n<|count|> 3"
    )
    perfect = scaffold_grpo_reward(
        gt,
        ground_truth_text=gt,
    )
    assert perfect["count_reward"] == 1.0, "perfect count should score 1"
    assert perfect["localization_reward"] == 1.0, "perfect localization should score 1"
    assert abs(perfect["format_reward"] - 0.2) < 1e-6, "well-formed output gets format bonus"

    wrong_count = scaffold_grpo_reward(
        "<|thought|>\nOops\n<|answer|>\n<|scaffold|> [[10,10],[30,30]]\n<|count|> 2",
        ground_truth_text=gt,
    )
    assert wrong_count["count_reward"] < 1.0, "wrong count should be penalized"

    bad_format = scaffold_grpo_reward("no count here", ground_truth_text=gt)
    assert bad_format["format_reward"] <= -0.5, "non-numeric output should be penalized"

    adapter_scores = scaffold_grpo_reward_func([gt], solution=[gt])
    assert len(adapter_scores) == 1 and adapter_scores[0] > 2.0, "adapter should return total reward"

    normalized = group_normalize_rewards([1.0, 2.0, 3.0])
    assert len(normalized) == 3 and normalized[0] < normalized[1] < normalized[2], "group normalization should preserve ordering"

    wrapped = reward_function(["prompt"], [gt], gt_points=[[[10, 10], [30, 30], [50, 50]]], gt_count=[3])
    assert len(wrapped) == 1 and wrapped[0] > 2.0, "reward wrapper should delegate to scaffold reward"

    crowd_reward = count_reward([[{"content": "<think>x</think><answer>3</answer>"}]], [3])
    assert crowd_reward and crowd_reward[0] >= 1.0, "CrowdVLM-style fuzzy count reward should be available"

    crowd_format = count_format_reward([[{"content": "<think>x</think><answer>3</answer>"}]])
    assert crowd_format == [1.0], "CrowdVLM-style format reward should match expected tag pattern"

    print("All Scaffold-R1 reward tests passed.")