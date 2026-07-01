"""
Instruction Template for Structured Decomposed Counting (Formulation 3)
------------------------------------------------------------------------
Builds the exact system + user prompt for feeding 4 quadrant crops to an MLLM.
Includes output parser and JSON schema validator.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────

@dataclass
class QuadrantMeta:
    name: str          # "top_left" etc.
    label: str         # "Image 1 — Top-Left"
    cut_edges: list    # interior cuts, e.g. ["RIGHT", "BOTTOM"]
    image_edges: list  # true image boundaries, e.g. ["TOP", "LEFT"]
    row: int           # 0-indexed grid row
    col: int           # 0-indexed grid col


@dataclass
class QuadrantCount:
    interior: int               = 0
    boundary_claimed: dict      = field(default_factory=dict)   # {edge: count}
    boundary_discarded: dict    = field(default_factory=dict)   # {edge: count}
    reasoning: str              = ""

    @property
    def total_claimed(self) -> int:
        return sum(self.boundary_claimed.values())

    @property
    def total_discarded(self) -> int:
        return sum(self.boundary_discarded.values())

    @property
    def subtotal(self) -> int:
        return self.interior + self.total_claimed


@dataclass
class StructuredCountOutput:
    quadrants: dict[str, QuadrantCount]   # keyed by "top_left" etc.
    predicted_total: int                  = 0
    sum_of_subtotals: int                 = 0
    raw_json: dict                        = field(default_factory=dict)
    parse_success: bool                   = True
    parse_error: str                      = ""


# ─────────────────────────────────────────────
#  Quadrant Layout Definition
# ─────────────────────────────────────────────

GRID_2x2 = [
    QuadrantMeta(
        name="top_left",
        label="Image 1 — Top-Left",
        cut_edges=["RIGHT", "BOTTOM"],
        image_edges=["TOP", "LEFT"],
        row=0, col=0
    ),
    QuadrantMeta(
        name="top_right",
        label="Image 2 — Top-Right",
        cut_edges=["LEFT", "BOTTOM"],
        image_edges=["TOP", "RIGHT"],
        row=0, col=1
    ),
    QuadrantMeta(
        name="bottom_left",
        label="Image 3 — Bottom-Left",
        cut_edges=["RIGHT", "TOP"],
        image_edges=["BOTTOM", "LEFT"],
        row=1, col=0
    ),
    QuadrantMeta(
        name="bottom_right",
        label="Image 4 — Bottom-Right",
        cut_edges=["LEFT", "TOP"],
        image_edges=["BOTTOM", "RIGHT"],
        row=1, col=1
    ),
]

# Shared edges: (quadrant_A, edge_A, quadrant_B, edge_B)
SHARED_EDGES_2x2 = [
    ("top_left",    "RIGHT",  "top_right",    "LEFT"),
    ("bottom_left", "RIGHT",  "bottom_right", "LEFT"),
    ("top_left",    "BOTTOM", "bottom_left",  "TOP"),
    ("top_right",   "BOTTOM", "bottom_right", "TOP"),
]


# ─────────────────────────────────────────────
#  JSON Output Schema
# ─────────────────────────────────────────────

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "counts", "consistency_check", "total"],
    "properties": {
        "reasoning": {
            "type": "object",
            "description": "Per-quadrant chain-of-thought before committing to counts",
            "required": ["top_left", "top_right", "bottom_left", "bottom_right"]
        },
        "counts": {
            "type": "object",
            "required": ["top_left", "top_right", "bottom_left", "bottom_right"],
            "additionalProperties": {
                "type": "object",
                "required": [
                    "interior",
                    "boundary_claimed_right", "boundary_discarded_right",
                    "boundary_claimed_bottom", "boundary_discarded_bottom",
                    "boundary_claimed_left",  "boundary_discarded_left",
                    "boundary_claimed_top",   "boundary_discarded_top",
                    "subtotal"
                ]
            }
        },
        "consistency_check": {
            "type": "object",
            "required": ["sum_of_subtotals", "note"]
        },
        "total": {"type": "integer"}
    }
}


# ─────────────────────────────────────────────
#  Prompt Builder
# ─────────────────────────────────────────────

class CountingPromptBuilder:
    """
    Builds the system + user prompt for structured decomposed counting.
    Images are passed separately as multimodal inputs to the MLLM.
    """

    def __init__(self, grid: list[QuadrantMeta] = GRID_2x2):
        self.grid = grid

    # ── System Prompt ──────────────────────────────────────────────────────────

    def build_system_prompt(self, category: str = "objects") -> str:
        quadrant_table = self._build_quadrant_table()
        edge_ownership_rules = self._build_edge_ownership_rules()
        output_format = self._build_output_format_spec()

        return f"""You are an expert object counter operating on image quadrants \
from a divide-and-conquer counting pipeline.

The input image has been divided into a {self._grid_shape()} non-overlapping grid. \
You will receive the four quadrant crops in reading order \
(top-left → top-right → bottom-left → bottom-right).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SPATIAL LAYOUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{quadrant_table}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
interior           : Objects whose centroid is clearly and fully inside this quadrant.
                     These are unambiguously owned by this quadrant.

boundary_claimed   : Objects that are VISUALLY CUT at a CUT EDGE, AND whose centroid
                     appears to be on THIS side of that cut. Count these here.

boundary_discarded : Objects that are VISUALLY CUT at a CUT EDGE, AND whose centroid
                     appears to be on the ADJACENT side of that cut. Do NOT count these;
                     they will be counted in the adjacent quadrant.

subtotal           : interior + sum(boundary_claimed across all cut edges)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EDGE OWNERSHIP RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{edge_ownership_rules}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CRITICAL CONSTRAINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every {category} in the full image must be claimed by EXACTLY ONE quadrant.
- An object claimed by quadrant A must be discarded by its adjacent quadrant.
- An object discarded by quadrant A must be claimed by its adjacent quadrant.
- No object may be claimed by two quadrants simultaneously.

Before finalizing your total, verify:
  sum(all subtotals) == total

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{output_format}

Respond ONLY with valid JSON. Do not include any text outside the JSON block.
Do not use markdown code fences."""

    # ── User Prompt ────────────────────────────────────────────────────────────

    def build_user_prompt(
        self,
        category: str = "objects",
        image_placeholders: bool = True,
        additional_context: str = ""
    ) -> str:
        """
        Returns the user-turn text. Actual image tensors/tokens are injected
        by the MLLM's multimodal interface at the <image_N> positions.
        """
        image_block = ""
        if image_placeholders:
            for i, quad in enumerate(self.grid, start=1):
                cut_str   = ", ".join(quad.cut_edges)
                image_str = ", ".join(quad.image_edges)
                image_block += (
                    f"<image_{i}>  "
                    f"[{quad.label} | cut edges: {cut_str} | image edges: {image_str}]\n"
                )
        else:
            image_block = "[Images provided as multimodal inputs in reading order]\n"

        context_block = f"\nAdditional context: {additional_context}\n" if additional_context else ""

        return f"""{image_block}{context_block}
Count all {category} across the full image.
Apply the centroid ownership rule strictly.
Provide the structured JSON output as specified."""

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _grid_shape(self) -> str:
        rows = max(q.row for q in self.grid) + 1
        cols = max(q.col for q in self.grid) + 1
        return f"{rows}×{cols}"

    def _build_quadrant_table(self) -> str:
        lines = []
        for i, q in enumerate(self.grid, start=1):
            cut_str   = ", ".join(q.cut_edges)   or "none"
            image_str = ", ".join(q.image_edges) or "none"
            lines.append(
                f"  Image {i}  →  {q.name.replace('_',' ').upper():<16} "
                f"cut edges: [{cut_str:<14}]  image edges: [{image_str}]"
            )
        return "\n".join(lines)

    def _build_edge_ownership_rules(self) -> str:
        rules = [
            "  • An object is CUT if it appears visually truncated at a quadrant edge.",
            "  • For each cut object, estimate where its CENTER (centroid) lies.",
            "  • Claim  it if the center is on YOUR side of the cut edge.",
            "  • Discard it if the center is on the ADJACENT quadrant's side.",
            "  • Objects cut at IMAGE EDGES (true borders) are still counted —",
            "    they are not duplicated anywhere.",
            "  • When in doubt about centroid position, use the majority-visible rule:",
            "    claim if more than half the object area is visible in this quadrant.",
        ]
        return "\n".join(rules)

    def _build_output_format_spec(self) -> str:
        quad_count_template = ""
        for q in self.grid:
            cut_fields = ""
            for edge in ["RIGHT", "BOTTOM", "LEFT", "TOP"]:
                if edge in q.cut_edges:
                    cut_fields += f'\n          "boundary_claimed_{edge.lower()}":   <int>,  // centroid here'
                    cut_fields += f'\n          "boundary_discarded_{edge.lower()}": <int>,  // centroid in adjacent'
                else:
                    cut_fields += f'\n          "boundary_claimed_{edge.lower()}":   0,      // image edge, no adjacent'
                    cut_fields += f'\n          "boundary_discarded_{edge.lower()}": 0,      // image edge, no adjacent'

            quad_count_template += f"""
      "{q.name}": {{
          "interior": <int>,  // fully-centered objects{cut_fields}
          "subtotal": <int>   // interior + sum of all boundary_claimed
      }},"""

        return f"""{{
  "reasoning": {{
      "top_left":     "<think aloud: interior count, then per-cut-edge boundary analysis>",
      "top_right":    "<think aloud>",
      "bottom_left":  "<think aloud>",
      "bottom_right": "<think aloud>"
  }},
  "counts": {{{quad_count_template}
  }},
  "consistency_check": {{
      "sum_of_subtotals": <int>,
      "note": "<flag any edge where claimed+discarded totals don't match across quadrants>"
  }},
  "total": <int>
}}"""

    # ── Full Prompt Package ────────────────────────────────────────────────────

    def build(self, category: str = "objects", additional_context: str = "") -> dict:
        """Returns the complete prompt as a dict ready for MLLM consumption."""
        return {
            "system": self.build_system_prompt(category),
            "user":   self.build_user_prompt(category, additional_context=additional_context),
            "schema": OUTPUT_SCHEMA,
        }


# ─────────────────────────────────────────────
#  Output Parser
# ─────────────────────────────────────────────

class StructuredOutputParser:
    """
    Parses the MLLM's JSON response into a StructuredCountOutput.
    Handles common failure modes: wrapped JSON, partial output, wrong types.
    """

    EDGES = ["right", "bottom", "left", "top"]

    def parse(self, raw_text: str) -> StructuredCountOutput:
        json_str = self._extract_json(raw_text)
        if json_str is None:
            return StructuredCountOutput(
                quadrants={},
                parse_success=False,
                parse_error="No valid JSON block found in model output"
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return StructuredCountOutput(
                quadrants={},
                parse_success=False,
                parse_error=f"JSON decode error: {e}"
            )

        return self._build_structured_output(data)

    def _extract_json(self, text: str) -> Optional[str]:
        # Try direct parse first
        try:
            json.loads(text.strip())
            return text.strip()
        except Exception:
            pass

        # Strip markdown fences
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)

        # Find outermost braces
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]

        return None

    def _build_structured_output(self, data: dict) -> StructuredCountOutput:
        quad_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        quadrants  = {}

        counts_block = data.get("counts", {})

        for name in quad_names:
            q_data = counts_block.get(name, {})
            claimed   = {}
            discarded = {}
            for edge in self.EDGES:
                claimed[edge]   = int(q_data.get(f"boundary_claimed_{edge}",   0))
                discarded[edge] = int(q_data.get(f"boundary_discarded_{edge}", 0))

            quadrants[name] = QuadrantCount(
                interior           = int(q_data.get("interior", 0)),
                boundary_claimed   = claimed,
                boundary_discarded = discarded,
                reasoning          = data.get("reasoning", {}).get(name, "")
            )

        predicted_total   = int(data.get("total", 0))
        sum_of_subtotals  = sum(q.subtotal for q in quadrants.values())

        return StructuredCountOutput(
            quadrants        = quadrants,
            predicted_total  = predicted_total,
            sum_of_subtotals = sum_of_subtotals,
            raw_json         = data,
            parse_success    = True
        )


# ─────────────────────────────────────────────
#  Quick Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":

    builder = CountingPromptBuilder()
    prompt  = builder.build(category="people")

    print("=" * 70)
    print("SYSTEM PROMPT")
    print("=" * 70)
    print(prompt["system"])

    print("\n" + "=" * 70)
    print("USER PROMPT")
    print("=" * 70)
    print(prompt["user"])

    # ── Simulate a model response and parse it ──
    mock_response = """
{
  "reasoning": {
    "top_left":     "I count 12 fully centered people. At the right edge I see 3 cut figures; 2 appear majority-visible here (centroid left of cut), 1 appears majority in top-right. At the bottom edge I see 2 cut figures, both with centers here.",
    "top_right":    "10 interior people. At the left edge 3 cut figures, 1 with center here, 2 belonging to top-left. At the bottom 2 cut figures, both centered here.",
    "bottom_left":  "11 interior. Right edge: 2 cut, 1 claimed. Top edge: 2 cut, 0 claimed (both centered in top-left).",
    "bottom_right": "9 interior. Left edge: 2 cut, 1 claimed. Top edge: 2 cut, 1 claimed."
  },
  "counts": {
    "top_left":     { "interior": 12, "boundary_claimed_right": 2,  "boundary_discarded_right": 1,  "boundary_claimed_bottom": 2, "boundary_discarded_bottom": 0, "boundary_claimed_left": 0, "boundary_discarded_left": 0, "boundary_claimed_top": 0, "boundary_discarded_top": 0, "subtotal": 16 },
    "top_right":    { "interior": 10, "boundary_claimed_right": 0,  "boundary_discarded_right": 0,  "boundary_claimed_bottom": 2, "boundary_discarded_bottom": 0, "boundary_claimed_left": 1, "boundary_discarded_left": 2, "boundary_claimed_top": 0, "boundary_discarded_top": 0, "subtotal": 13 },
    "bottom_left":  { "interior": 11, "boundary_claimed_right": 1,  "boundary_discarded_right": 1,  "boundary_claimed_bottom": 0, "boundary_discarded_bottom": 0, "boundary_claimed_left": 0, "boundary_discarded_left": 0, "boundary_claimed_top": 0, "boundary_discarded_top": 2, "subtotal": 12 },
    "bottom_right": { "interior": 9,  "boundary_claimed_right": 0,  "boundary_discarded_right": 0,  "boundary_claimed_bottom": 0, "boundary_discarded_bottom": 0, "boundary_claimed_left": 1, "boundary_discarded_left": 1, "boundary_claimed_top": 1, "boundary_discarded_top": 1, "subtotal": 11 }
  },
  "consistency_check": {
    "sum_of_subtotals": 52,
    "note": "All edge totals appear balanced."
  },
  "total": 52
}
"""

    parser = StructuredOutputParser()
    output = parser.parse(mock_response)

    print("\n" + "=" * 70)
    print("PARSED OUTPUT")
    print("=" * 70)
    if output.parse_success:
        for name, qc in output.quadrants.items():
            print(f"  {name:<15}  interior={qc.interior}  "
                  f"claimed={qc.total_claimed}  discarded={qc.total_discarded}  "
                  f"subtotal={qc.subtotal}")
        print(f"\n  Sum of subtotals : {output.sum_of_subtotals}")
        print(f"  Predicted total  : {output.predicted_total}")
        print(f"  Coherent         : {output.sum_of_subtotals == output.predicted_total}")
    else:
        print(f"  Parse failed: {output.parse_error}")