"""
Reward 3: Direct Boundary Overcounting Penalty
-----------------------------------------------
Detects and quantifies objects claimed by both adjacent quadrants simultaneously.
No GT annotations required — purely structural / self-verifiable.

Includes:
  - BoundaryOvercountingReward  : the core R3 reward
  - SumCoherenceReward          : R_sum (arithmetic consistency)
  - EdgeConservationReward      : R_conservation (same edge seen from both sides)
  - CompositeReward             : weighted combination for GRPO
  - GRPORewardWrapper           : drop-in for GRPO rollout scoring
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from prompt_template import (
    StructuredCountOutput, QuadrantCount, SHARED_EDGES_2x2
)


# ─────────────────────────────────────────────
#  Reward Result Container
# ─────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    # Per-edge overcounting
    overcount_per_edge: dict = field(default_factory=dict)   # {edge_key: int}
    total_overcount: int = 0

    # Per-edge conservation violation
    conservation_per_edge: dict = field(default_factory=dict)
    total_conservation_violation: float = 0.0

    # Arithmetic coherence
    sum_declared: int = 0
    total_declared: int = 0
    sum_coherence_error: int = 0

    # Final reward scalars (all ≤ 0; 0 is perfect)
    R3_overcount: float = 0.0
    R_conservation: float = 0.0
    R_sum: float = 0.0
    R_composite: float = 0.0

    def __str__(self):
        lines = [
            "── Reward Breakdown ─────────────────────────────",
            f"  R3  Boundary overcounting  : {self.R3_overcount:+.3f}   "
            f"(raw overcount = {self.total_overcount} objects)",
        ]
        if self.total_overcount > 0:
            for edge_key, val in self.overcount_per_edge.items():
                if val > 0:
                    lines.append(f"        ↳ {edge_key:<35} +{val}")
        lines += [
            f"  R_c Edge conservation       : {self.R_conservation:+.3f}   "
            f"(total violation = {self.total_conservation_violation:.1f})",
            f"  R_s Sum coherence           : {self.R_sum:+.3f}   "
            f"(declared {self.total_declared}, sum {self.sum_declared})",
            f"  ─────────────────────────────────────────────",
            f"  R_composite                 : {self.R_composite:+.3f}",
            "─────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Reward 3: Direct Boundary Overcounting
# ─────────────────────────────────────────────

class BoundaryOvercountingReward:
    """
    For each shared cut edge between two adjacent quadrants A and B:

        c_A = boundary_claimed by A at that edge
        d_A = boundary_discarded by A at that edge
        c_B = boundary_claimed by B at that edge
        d_B = boundary_discarded by B at that edge

    Perfect assignment:  c_A + c_B == c_A + d_A  (== c_B + d_B == N_edge)
                         i.e.,  c_B == d_A  AND  c_A == d_B

    Overcounting excess: max(0, c_B - d_A)   ← B claims more than A discarded
                       + max(0, c_A - d_B)   ← A claims more than B discarded

    Interpretation: each unit of excess = one object counted in BOTH quadrants.

    The reward is  R3 = -scale * total_overcount_excess
    where scale controls sensitivity relative to other reward components.
    """

    def __init__(
        self,
        scale: float = 1.0,
        normalize_by_total: bool = False,
        shared_edges: list = None
    ):
        self.scale = scale
        self.normalize_by_total = normalize_by_total
        self.shared_edges = shared_edges or SHARED_EDGES_2x2

    def compute(
        self,
        output: StructuredCountOutput,
        gt_total: Optional[int] = None
    ) -> tuple[float, dict]:
        """
        Returns (reward_scalar, per_edge_breakdown).

        reward_scalar ≤ 0. 0 means no double-counting detected.
        per_edge_breakdown: {edge_key: excess_count}
        """
        if not output.parse_success or not output.quadrants:
            return -999.0, {}   # parse failure → very large penalty

        per_edge = {}
        total_excess = 0

        for (quad_a, edge_a, quad_b, edge_b) in self.shared_edges:
            qc_a: QuadrantCount = output.quadrants.get(quad_a)
            qc_b: QuadrantCount = output.quadrants.get(quad_b)

            if qc_a is None or qc_b is None:
                continue

            edge_a_lo = edge_a.lower()
            edge_b_lo = edge_b.lower()

            c_A = qc_a.boundary_claimed.get(edge_a_lo, 0)
            d_A = qc_a.boundary_discarded.get(edge_a_lo, 0)
            c_B = qc_b.boundary_claimed.get(edge_b_lo, 0)
            d_B = qc_b.boundary_discarded.get(edge_b_lo, 0)

            # Excess claims on each side beyond what the other side discarded
            excess_from_B = max(0, c_B - d_A)   # B over-claimed
            excess_from_A = max(0, c_A - d_B)   # A over-claimed
            edge_excess   = excess_from_B + excess_from_A

            edge_key = f"{quad_a}[{edge_a}] ↔ {quad_b}[{edge_b}]"
            per_edge[edge_key] = {
                "c_A": c_A, "d_A": d_A,
                "c_B": c_B, "d_B": d_B,
                "excess_from_A": excess_from_A,
                "excess_from_B": excess_from_B,
                "total_excess":  edge_excess
            }
            total_excess += edge_excess

        # Optional: normalize excess by predicted total so scale is relative
        if self.normalize_by_total and output.predicted_total > 0:
            reward = -self.scale * (total_excess / output.predicted_total)
        else:
            reward = -self.scale * total_excess

        return reward, per_edge


# ─────────────────────────────────────────────
#  Edge Conservation Reward
# ─────────────────────────────────────────────

class EdgeConservationReward:
    """
    For each shared edge, the total number of boundary objects visible from
    both sides should be equal:

        (c_A + d_A)  ==  (c_B + d_B)  ==  N_edge

    Violation means the two quadrants disagree on how many objects exist at
    that edge — a structural impossibility.

    R_conservation = -scale * sum_of_absolute_violations
    """

    def __init__(self, scale: float = 1.0, shared_edges: list = None):
        self.scale = scale
        self.shared_edges = shared_edges or SHARED_EDGES_2x2

    def compute(self, output: StructuredCountOutput) -> tuple[float, dict]:
        if not output.parse_success or not output.quadrants:
            return -999.0, {}

        per_edge = {}
        total_violation = 0.0

        for (quad_a, edge_a, quad_b, edge_b) in self.shared_edges:
            qc_a = output.quadrants.get(quad_a)
            qc_b = output.quadrants.get(quad_b)
            if qc_a is None or qc_b is None:
                continue

            edge_a_lo = edge_a.lower()
            edge_b_lo = edge_b.lower()

            total_A = (qc_a.boundary_claimed.get(edge_a_lo, 0)
                     + qc_a.boundary_discarded.get(edge_a_lo, 0))
            total_B = (qc_b.boundary_claimed.get(edge_b_lo, 0)
                     + qc_b.boundary_discarded.get(edge_b_lo, 0))

            violation = abs(total_A - total_B)
            edge_key  = f"{quad_a}[{edge_a}] ↔ {quad_b}[{edge_b}]"
            per_edge[edge_key] = {
                "total_A": total_A, "total_B": total_B, "violation": violation
            }
            total_violation += violation

        reward = -self.scale * total_violation
        return reward, per_edge


# ─────────────────────────────────────────────
#  Sum Coherence Reward
# ─────────────────────────────────────────────

class SumCoherenceReward:
    """
    The declared 'total' must equal the arithmetic sum of all quadrant subtotals.
    R_sum = -scale * |sum_of_subtotals - declared_total|
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def compute(self, output: StructuredCountOutput) -> float:
        if not output.parse_success:
            return -999.0
        error = abs(output.sum_of_subtotals - output.predicted_total)
        return -self.scale * error


# ─────────────────────────────────────────────
#  Outcome Reward (when GT available)
# ─────────────────────────────────────────────

class OutcomeReward:
    """
    Asymmetric reward: overcounting penalized more heavily than undercounting.
    R_outcome = -alpha * error  if predicted > gt   (overcounting)
              = -error          if predicted <= gt  (undercounting)
    """

    def __init__(self, alpha: float = 2.0, scale: float = 1.0):
        self.alpha = alpha
        self.scale = scale

    def compute(self, output: StructuredCountOutput, gt_total: int) -> float:
        if not output.parse_success:
            return -999.0
        error = output.predicted_total - gt_total
        if error > 0:
            return -self.scale * self.alpha * error   # overcounting: heavier
        else:
            return -self.scale * abs(error)           # undercounting: standard


# ─────────────────────────────────────────────
#  Composite Reward
# ─────────────────────────────────────────────

@dataclass
class RewardWeights:
    R3_overcount:   float = 0.40   # primary: direct boundary double-counting
    R_conservation: float = 0.25   # structural edge consistency
    R_sum:          float = 0.20   # arithmetic coherence
    R_outcome:      float = 0.15   # outcome signal (only when GT available)

    def validate(self):
        total = self.R3_overcount + self.R_conservation + self.R_sum + self.R_outcome
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


class CompositeReward:
    """
    Combines all reward components. R_outcome weight is redistributed
    proportionally to structural rewards when GT is not available.
    """

    def __init__(self, weights: RewardWeights = None):
        self.weights  = weights or RewardWeights()
        self.r3       = BoundaryOvercountingReward(scale=1.0)
        self.r_cons   = EdgeConservationReward(scale=1.0)
        self.r_sum    = SumCoherenceReward(scale=1.0)
        self.r_out    = OutcomeReward(alpha=2.0, scale=1.0)

    def compute(
        self,
        output: StructuredCountOutput,
        gt_total: Optional[int] = None
    ) -> RewardBreakdown:

        r3_scalar,   edge_overcount  = self.r3.compute(output, gt_total)
        r_c_scalar,  edge_conserve   = self.r_cons.compute(output)
        r_s_scalar                   = self.r_sum.compute(output)

        total_overcount = sum(
            v["total_excess"] for v in edge_overcount.values()
        ) if edge_overcount else 0

        total_conservation = sum(
            v["violation"] for v in edge_conserve.values()
        ) if edge_conserve else 0.0

        w = self.weights

        if gt_total is not None:
            r_out_scalar = self.r_out.compute(output, gt_total)
            composite = (
                w.R3_overcount   * r3_scalar   +
                w.R_conservation * r_c_scalar  +
                w.R_sum          * r_s_scalar  +
                w.R_outcome      * r_out_scalar
            )
        else:
            # Redistribute outcome weight to structural rewards
            total_structural = w.R3_overcount + w.R_conservation + w.R_sum
            composite = (
                (w.R3_overcount   / total_structural) * r3_scalar  +
                (w.R_conservation / total_structural) * r_c_scalar +
                (w.R_sum          / total_structural) * r_s_scalar
            )
            r_out_scalar = 0.0

        return RewardBreakdown(
            overcount_per_edge          = {k: v["total_excess"] for k, v in edge_overcount.items()},
            total_overcount             = total_overcount,
            conservation_per_edge       = edge_conserve,
            total_conservation_violation= total_conservation,
            sum_declared                = output.sum_of_subtotals,
            total_declared              = output.predicted_total,
            sum_coherence_error         = abs(output.sum_of_subtotals - output.predicted_total),
            R3_overcount                = r3_scalar,
            R_conservation              = r_c_scalar,
            R_sum                       = r_s_scalar,
            R_composite                 = composite,
        )


# ─────────────────────────────────────────────
#  GRPO Rollout Scorer
# ─────────────────────────────────────────────

class GRPORewardWrapper:
    """
    Drop-in scorer for GRPO rollout groups.

    Usage:
        scorer = GRPORewardWrapper(gt_totals={"img_001": 47, ...})

        # For each rollout group (K samples for the same image):
        rewards = scorer.score_group(
            image_id   = "img_001",
            raw_outputs = [model_out_1, model_out_2, ..., model_out_K]
        )
        advantages = scorer.normalize_advantages(rewards)
    """

    def __init__(
        self,
        gt_totals: Optional[dict[str, int]] = None,
        weights:   Optional[RewardWeights]  = None,
        parser_cls = None
    ):
        from prompt_template import StructuredOutputParser
        self.gt_totals = gt_totals or {}
        self.reward    = CompositeReward(weights)
        self.parser    = (parser_cls or StructuredOutputParser)()

    def score_group(
        self,
        image_id: str,
        raw_outputs: list[str]
    ) -> list[float]:
        gt = self.gt_totals.get(image_id)
        rewards = []
        for raw in raw_outputs:
            parsed   = self.parser.parse(raw)
            breakdown = self.reward.compute(parsed, gt_total=gt)
            rewards.append(breakdown.R_composite)
        return rewards

    def normalize_advantages(
        self,
        rewards: list[float],
        eps: float = 1e-8
    ) -> list[float]:
        """Group-relative advantage normalization (standard GRPO)."""
        mean_r = sum(rewards) / len(rewards)
        std_r  = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards))
        return [(r - mean_r) / (std_r + eps) for r in rewards]

    def score_and_normalize(
        self,
        image_id: str,
        raw_outputs: list[str]
    ) -> tuple[list[float], list[float]]:
        rewards    = self.score_group(image_id, raw_outputs)
        advantages = self.normalize_advantages(rewards)
        return rewards, advantages


# ─────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from prompt_template import StructuredOutputParser

    parser  = StructuredOutputParser()
    reward  = CompositeReward()

    # ── Case 1: Perfect assignment (no overcounting) ──────────────────────────
    perfect_output = """{
      "reasoning": {"top_left":"ok","top_right":"ok","bottom_left":"ok","bottom_right":"ok"},
      "counts": {
        "top_left":     {"interior":12,"boundary_claimed_right":2,"boundary_discarded_right":1,
                         "boundary_claimed_bottom":2,"boundary_discarded_bottom":0,
                         "boundary_claimed_left":0,"boundary_discarded_left":0,
                         "boundary_claimed_top":0,"boundary_discarded_top":0,"subtotal":16},
        "top_right":    {"interior":10,"boundary_claimed_right":0,"boundary_discarded_right":0,
                         "boundary_claimed_bottom":1,"boundary_discarded_bottom":1,
                         "boundary_claimed_left":1,"boundary_discarded_left":2,
                         "boundary_claimed_top":0,"boundary_discarded_top":0,"subtotal":12},
        "bottom_left":  {"interior":11,"boundary_claimed_right":1,"boundary_discarded_right":1,
                         "boundary_claimed_bottom":0,"boundary_discarded_bottom":0,
                         "boundary_claimed_left":0,"boundary_discarded_left":0,
                         "boundary_claimed_top":0,"boundary_discarded_top":2,"subtotal":12},
        "bottom_right": {"interior":9, "boundary_claimed_right":0,"boundary_discarded_right":0,
                         "boundary_claimed_bottom":0,"boundary_discarded_bottom":0,
                         "boundary_claimed_left":1,"boundary_discarded_left":1,
                         "boundary_claimed_top":1,"boundary_discarded_top":1,"subtotal":11}
      },
      "consistency_check": {"sum_of_subtotals":51,"note":"balanced"},
      "total": 51
    }"""

    # ── Case 2: Boundary overcounting (both TL and TR claim right/left edge) ──
    overcounting_output = """{
      "reasoning": {"top_left":"ok","top_right":"ok","bottom_left":"ok","bottom_right":"ok"},
      "counts": {
        "top_left":     {"interior":12,"boundary_claimed_right":3,"boundary_discarded_right":0,
                         "boundary_claimed_bottom":2,"boundary_discarded_bottom":0,
                         "boundary_claimed_left":0,"boundary_discarded_left":0,
                         "boundary_claimed_top":0,"boundary_discarded_top":0,"subtotal":17},
        "top_right":    {"interior":10,"boundary_claimed_right":0,"boundary_discarded_right":0,
                         "boundary_claimed_bottom":1,"boundary_discarded_bottom":1,
                         "boundary_claimed_left":3,"boundary_discarded_left":0,
                         "boundary_claimed_top":0,"boundary_discarded_top":0,"subtotal":14},
        "bottom_left":  {"interior":11,"boundary_claimed_right":1,"boundary_discarded_right":1,
                         "boundary_claimed_bottom":0,"boundary_discarded_bottom":0,
                         "boundary_claimed_left":0,"boundary_discarded_left":0,
                         "boundary_claimed_top":0,"boundary_discarded_top":2,"subtotal":12},
        "bottom_right": {"interior":9, "boundary_claimed_right":0,"boundary_discarded_right":0,
                         "boundary_claimed_bottom":0,"boundary_discarded_bottom":0,
                         "boundary_claimed_left":1,"boundary_discarded_left":1,
                         "boundary_claimed_top":1,"boundary_discarded_top":1,"subtotal":11}
      },
      "consistency_check": {"sum_of_subtotals":54,"note":"might be high"},
      "total": 54
    }"""

    gt = 51

    for label, raw in [("PERFECT ASSIGNMENT", perfect_output),
                        ("BOUNDARY OVERCOUNTING", overcounting_output)]:
        parsed   = parser.parse(raw)
        breakdown = reward.compute(parsed, gt_total=gt)

        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")
        print(breakdown)

    # ── GRPO Group Simulation ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  GRPO GROUP SCORING (K=4 rollouts, GT=51)")
    print("="*55)

    scorer = GRPORewardWrapper(gt_totals={"img_001": 51})
    rollouts = [perfect_output, overcounting_output,
                overcounting_output, perfect_output]

    rewards, advantages = scorer.score_and_normalize("img_001", rollouts)

    for i, (r, a) in enumerate(zip(rewards, advantages)):
        print(f"  Rollout {i+1}:  reward={r:+.4f}   advantage={a:+.4f}")

    print("\n  → Positive advantage → model is encouraged toward this output")
    print("  → Negative advantage → model is discouraged from this output")