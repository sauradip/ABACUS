"""
Local CPU/GPU-free debug test for stability-override dtype correctness.

Defines _wrap_fp32_forward and apply_stability_overrides inline (copies from
generate_scaffold_rex_audit_v2.py) so we can run without PIL/transformers/peft.

Run with:
    .venv311/bin/python scripts/counting_grpo/debug_dtype_audit.py
"""
import sys
import torch
import torch.nn as nn
from typing import Any


# ---------------------------------------------------------------------------
# Inline copies of the two helpers — kept in sync with the audit script.
# If you update one, update both.
# ---------------------------------------------------------------------------

def _wrap_fp32_forward(module: nn.Module) -> None:
    """Wrap forward pass to cast input through float32 for numerical stability."""
    if getattr(module, "_stage15_fp32_wrapped", False):
        return
    original_forward = module.forward

    def _forward_cast(x: Any, *args: Any, **kwargs: Any) -> Any:
        input_dtype = x.dtype if torch.is_tensor(x) else None
        if torch.is_tensor(x):
            x = x.float()
        out = original_forward(x, *args, **kwargs)
        if input_dtype is not None and torch.is_tensor(out):
            return out.to(input_dtype)
        return out

    module.forward = _forward_cast
    module._stage15_fp32_wrapped = True


def apply_stability_overrides(model: nn.Module, vision_scale: float = 1.0) -> None:
    """Apply Stage 1.5 stability overrides: FP32 RMSNorm casting + vision scaling."""
    output_embeddings = None
    if hasattr(model.language_model, "get_output_embeddings"):
        output_embeddings = model.language_model.get_output_embeddings()
    if output_embeddings is None and hasattr(model.language_model, "output"):
        output_embeddings = model.language_model.output
    # NOTE: do NOT cast output_embeddings to float32.
    # The RMSNorm wrapper restores bfloat16 on output, so the LM head
    # must also stay bfloat16 at inference (unlike training which has autocast).

    for _, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if "rmsnorm" in class_name:
            module.to(torch.float32)
            _wrap_fp32_forward(module)

    if hasattr(model, "mlp1") and hasattr(model.mlp1, "forward"):
        if not getattr(model.mlp1, "_stage15_scaled", False):
            original_forward = model.mlp1.forward

            def _scaled_forward(*args: Any, **kwargs: Any) -> Any:
                return original_forward(*args, **kwargs) * vision_scale

            model.mlp1.forward = _scaled_forward
            model.mlp1._stage15_scaled = True

    print(f"[info] Applied stability overrides: vision_scale={vision_scale}")

# ---------------------------------------------------------------------------
# Mock model mirroring the real InternVL2 structure at a tiny scale
# ---------------------------------------------------------------------------
D = 16   # hidden dim

class MockRMSNorm(nn.Module):
    """Minimal layer-norm that stands in for InternLM2RMSNorm."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        return x * self.weight


class MockLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm  = MockRMSNorm()
        self.output = nn.Linear(D, 64, bias=False)

    def get_output_embeddings(self):
        return self.output


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = MockLanguageModel()
        self.mlp1 = nn.Linear(D, D, bias=False)


def run_tests():
    print("=== debug_dtype_audit.py ===")

    # Build model, put everything in bfloat16 (mirrors inference state)
    model = MockModel().to(torch.bfloat16)

    # Apply the stability overrides with vision_scale=1.0
    apply_stability_overrides(model, vision_scale=1.0)

    # ------------------------------------------------------------------
    # Test 1: RMSNorm dtype round-trip
    # ------------------------------------------------------------------
    x = torch.randn(2, D, dtype=torch.bfloat16)
    out = model.language_model.norm(x)
    assert out.dtype == torch.bfloat16, f"FAIL T1: norm output dtype={out.dtype}, expected bfloat16"
    print("PASS T1: RMSNorm forward stays bfloat16 →", out.dtype)

    # ------------------------------------------------------------------
    # Test 2: LM head (output linear) dtype compatibility
    # ------------------------------------------------------------------
    lm_head = model.language_model.output
    print(f"      LM head weight dtype: {lm_head.weight.dtype}")
    try:
        logits = lm_head(out)
        assert logits.dtype == torch.bfloat16, f"FAIL T2: logits dtype={logits.dtype}"
        print("PASS T2: LM head forward (no dtype mismatch) →", logits.dtype)
    except RuntimeError as e:
        print(f"FAIL T2: dtype mismatch in LM head: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Test 3: mlp1 vision_scale wrapper
    # ------------------------------------------------------------------
    assert getattr(model.mlp1, "_stage15_scaled", False), "FAIL T3: mlp1 not scaled"
    v = torch.randn(2, D, dtype=torch.bfloat16)
    scaled = model.mlp1(v)
    assert scaled.dtype == torch.bfloat16, f"FAIL T3: mlp1 output dtype={scaled.dtype}"
    print("PASS T3: mlp1 vision_scale wrapper preserves bfloat16 →", scaled.dtype)

    # ------------------------------------------------------------------
    # Test 4: output_embeddings NOT cast to float32
    # ------------------------------------------------------------------
    assert lm_head.weight.dtype == torch.bfloat16, (
        f"FAIL T4: LM head weight was cast to float32 — should stay bfloat16. "
        f"Got {lm_head.weight.dtype}"
    )
    print("PASS T4: LM head weight remained bfloat16 →", lm_head.weight.dtype)

    print("\n✅ All dtype tests passed — safe to submit to SLURM.")


if __name__ == "__main__":
    run_tests()
