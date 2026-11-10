import pytest

from exp593_dna.anneal import OWNER, TRIALS
from exp593_dna.anneal_final_audit import AuditConfig


def test_native_audit_identifies_exact_normal_and_retry_checkpoint():
    prefix = f"{OWNER}/annealing/training/{TRIALS['181b'].run_id}/checkpoints/step-"
    regular = AuditConfig("181b", prefix + "356535", "a" * 64, "cw-rno2a", True)
    retry = AuditConfig(
        "181b", prefix + "356535-retry-" + "b" * 32, "a" * 64, "cw-rno2a", True
    )
    assert regular.checkpoint_parts == (356535, None)
    assert regular.identity == "dna-exp593-anneal-nativeaudit-181b-step-356535-v1"
    assert retry.checkpoint_parts == (356535, "b" * 32)
    assert retry.identity == (
        "dna-exp593-anneal-nativeaudit-181b-step-356535-retry-bbbbbbbb-v1"
    )
    with pytest.raises(ValueError, match="invalid step/retry suffix"):
        AuditConfig("181b", prefix + "356535-retry-bad", "a" * 64, "cw-rno2a")
    with pytest.raises(ValueError, match="escaped the production namespace"):
        AuditConfig(
            "181b",
            prefix.replace("training/", "validation/") + "356535",
            "a" * 64,
            "cw-rno2a",
        )
