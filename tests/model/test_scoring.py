"""Tests for ``marin_dna.model.scoring`` (CLM-only).

Vendored from biofoundation/tests/test_scoring.py at commit 834dd4c (May 2026).
MLM test (``test_run_llr_mlm_rc_avg_equals_mean_of_two_passes``) dropped —
marin-dna's vendored scoring module is CLM-only.

The helper test doubles (``_DeterministicCLM``,
``_DeterministicCausalLMWithEmbeddings``) are plain ``nn.Module`` subclasses
here. After the migration, ``marin_dna.model.scoring`` calls
``model(input_ids).logits`` (and ``output.hidden_states[i]`` when
``output_hidden_states=True``) directly — no ``CausalLM`` /
``CausalLMWithEmbeddings`` abstract bases. The helpers return
``SimpleNamespace`` objects exposing the same attribute surface as HF
``CausalLMOutput``.
"""

import math
from functools import partial
from types import SimpleNamespace

import datasets
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.data.genome import Genome
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    transform_llr_clm,
    transform_reflogprob_clm,
)
from marin_dna.model.runner import (
    run_inference,
    run_ll_clm,
    run_variant_score_bundle,
)
from marin_dna.model.scoring import (
    _logits_to_logprobs,
    compute_ll_clm,
    compute_reflogprob_clm,
    compute_variant_score_bundle,
)


TINY_CLM = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"


def _load_tiny_clm():
    return AutoModelForCausalLM.from_pretrained(TINY_CLM)


class _DeterministicCLM(nn.Module):
    """Test double whose forward returns a fixed logits tensor wrapped in an
    HF-style output object (``.logits``)."""

    def __init__(self, logits: Tensor):
        super().__init__()
        # Register as buffer so .to(device) works, but value is fixed.
        self.register_buffer("_logits", logits)

    def forward(self, input_ids):
        # Returns a *copy* sliced to the input batch/length so callers can
        # use any input_ids shape that matches.
        B, L = input_ids.shape
        assert self._logits.shape[0] >= B and self._logits.shape[1] >= L
        return SimpleNamespace(logits=self._logits[:B, :L].clone())


def test_compute_ll_clm_matches_hf_cross_entropy():
    """ll_sum / n  ==  -model(input_ids, labels=input_ids).loss.

    HF's CausalLM models compute loss as mean cross-entropy over the L-1
    shifted targets. Dividing our per-row ll_sum by n recovers the same
    quantity, with the standard sign flip.
    """
    torch.manual_seed(0)
    model = _load_tiny_clm()
    raw = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    raw.eval()
    model.eval()

    vocab_size = raw.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (3, 17))

    with torch.no_grad():
        out = compute_ll_clm(model, input_ids)  # [B, 2]
        assert out.shape == (3, 2)
        ll_mean = out[:, 0] / out[:, 1]
        for i in range(input_ids.shape[0]):
            hf_loss = raw(input_ids[i : i + 1], labels=input_ids[i : i + 1]).loss
            assert math.isclose(
                ll_mean[i].item(), -hf_loss.item(), rel_tol=1e-5, abs_tol=1e-5
            ), f"row {i}: ours={ll_mean[i].item()} hf={-hf_loss.item()}"


def test_compute_ll_clm_hand_computed_two_token():
    """Smallest non-trivial off-by-one check with a known logits tensor."""
    # Vocab size 4, batch 1, length 3
    # logits[0, 0] predicts input_ids[0, 1]
    # logits[0, 1] predicts input_ids[0, 2]
    # logits[0, 2] is unused (last position)
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [9.0, 9.0, 9.0, 9.0],
            ]
        ]
    )
    input_ids = torch.tensor([[3, 1, 0]])
    model = _DeterministicCLM(logits)

    log_softmax_0 = torch.log_softmax(logits[0, 0], dim=-1)
    log_softmax_1 = torch.log_softmax(logits[0, 1], dim=-1)
    expected_sum = (log_softmax_0[1] + log_softmax_1[0]).item()

    out = compute_ll_clm(model, input_ids)
    assert out.shape == (1, 2)
    assert math.isclose(out[0, 0].item(), expected_sum, rel_tol=1e-6, abs_tol=1e-7)
    assert out[0, 1].item() == 2.0


def test_compute_ll_clm_target_side_shift():
    """is_upper applies to the *target* token (input_ids[i+1]), not the source."""
    torch.manual_seed(1)
    B, L, V = 1, 8, 5
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    # Source-aligned mask: positions [0..3] uppercase, [4..7] lowercase.
    is_upper = torch.tensor([[True, True, True, True, False, False, False, False]])

    out = compute_ll_clm(model, input_ids, is_upper)  # [B, 4]
    assert out.shape == (1, 4)
    ll_sum_upper, ll_sum_lower, n_upper, n_lower = out[0].tolist()

    # Manual computation
    log_softmax = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = input_ids[0, 1:]
    per_target_logp = log_softmax[torch.arange(L - 1), targets]
    # Target frame: is_upper[1:] = [T, T, T, F, F, F, F]  → 3 upper, 4 lower
    upper_mask = is_upper[0, 1:]
    lower_mask = ~upper_mask

    expected_sum_upper = per_target_logp[upper_mask].sum().item()
    expected_sum_lower = per_target_logp[lower_mask].sum().item()

    assert math.isclose(ll_sum_upper, expected_sum_upper, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(ll_sum_lower, expected_sum_lower, rel_tol=1e-6, abs_tol=1e-7)
    assert n_upper == 3.0
    assert n_lower == 4.0

    # Sanity: had we mistakenly used the SOURCE frame [:-1] — would give
    # n_upper=4, n_lower=3 (different counts) and a different sum.
    assert n_upper != is_upper[0, :-1].sum().item()


def test_compute_ll_clm_invariants():
    """Per-row invariants of the [B, 4] output."""
    torch.manual_seed(2)
    B, L, V = 4, 11, 7
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    is_upper = torch.zeros(B, L, dtype=torch.bool)
    is_upper[0, :3] = True
    is_upper[1, :7] = True
    is_upper[2, :2] = True
    is_upper[3, :9] = True

    out = compute_ll_clm(model, input_ids, is_upper)  # [B, 4]
    ll_sum_upper, ll_sum_lower, n_upper, n_lower = out.unbind(-1)
    out_no_mask = compute_ll_clm(model, input_ids)  # [B, 2]
    ll_sum_total, n_total = out_no_mask.unbind(-1)

    # Sums partition the total
    assert torch.allclose(ll_sum_upper + ll_sum_lower, ll_sum_total, atol=1e-5)
    # Counts partition L-1
    assert torch.equal(n_upper + n_lower, n_total)
    assert torch.all(n_total == float(L - 1))


def test_compute_ll_clm_dataset_wide_token_weighted_mean():
    """The intended aggregation pattern works and beats avg-of-means
    when n_upper / n_lower vary across rows."""
    torch.manual_seed(5)
    B, L, V = 4, 11, 7
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    is_upper = torch.zeros(B, L, dtype=torch.bool)
    is_upper[0, :3] = True
    is_upper[1, :7] = True
    is_upper[2, :2] = True
    is_upper[3, :9] = True

    out = compute_ll_clm(
        model, input_ids, is_upper
    ).double()  # cast for fp64 accumulate
    S_u, S_l, n_u, n_l = out.sum(dim=0).unbind(-1)
    LL_all = ((S_u + S_l) / (n_u + n_l)).item()
    LL_upper = (S_u / n_u).item()
    LL_lower = (S_l / n_l).item()

    # Brute-force: gather *every* target logp across the whole batch and
    # split by mask.
    log_softmax = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:]
    per_target_logp = torch.gather(log_softmax, 2, targets.unsqueeze(-1)).squeeze(-1)
    target_upper = is_upper[:, 1:]
    expected_LL_all = per_target_logp.mean().item()
    expected_LL_upper = per_target_logp[target_upper].mean().item()
    expected_LL_lower = per_target_logp[~target_upper].mean().item()

    assert math.isclose(LL_all, expected_LL_all, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(LL_upper, expected_LL_upper, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(LL_lower, expected_LL_lower, rel_tol=1e-6, abs_tol=1e-7)

    # Sanity that this differs from avg-of-per-sequence-means with
    # heterogeneous counts (the wrong way to aggregate).
    per_seq_upper = (out[:, 0] / out[:, 2]).double()
    naive = per_seq_upper.mean().item()
    assert not math.isclose(LL_upper, naive, rel_tol=1e-3, abs_tol=1e-3)


def test_compute_ll_clm_all_upper_or_all_lower_rows_aggregate_correctly():
    """All-upper / all-lower rows have n=0 in one bucket, ll_sum=0 there.
    They still contribute correctly when summing across the dataset (no
    NaN in the per-row tensor, no NaN gymnastics needed at aggregation)."""
    torch.manual_seed(8)
    B, L, V = 3, 6, 4
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    is_upper = torch.zeros(B, L, dtype=torch.bool)
    is_upper[0, :] = True  # row 0: all upper (target frame too)
    # row 1: all lower (default)
    is_upper[2, :3] = True  # row 2: mixed

    out = compute_ll_clm(model, input_ids, is_upper)
    assert out.shape == (B, 4)
    # Per-row primitive output is finite everywhere — no NaN to manage.
    assert torch.isfinite(out).all()
    # Row 0: n_lower == 0, ll_sum_lower == 0
    assert out[0, 1].item() == 0.0
    assert out[0, 3].item() == 0.0
    # Row 1: n_upper == 0, ll_sum_upper == 0
    assert out[1, 0].item() == 0.0
    assert out[1, 2].item() == 0.0

    # Aggregating across all 3 rows still gives meaningful global LLs
    S_u, S_l, n_u, n_l = out.double().sum(dim=0).tolist()
    assert n_u > 0 and n_l > 0  # because row 2 contributes to both
    LL_upper = S_u / n_u
    LL_lower = S_l / n_l
    assert math.isfinite(LL_upper) and math.isfinite(LL_lower)


def test_compute_ll_clm_shape_without_mask():
    torch.manual_seed(4)
    B, L, V = 2, 6, 4
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)
    out = compute_ll_clm(model, input_ids)
    assert out.shape == (B, 2)
    assert torch.all(out[:, 1] == float(L - 1))


def test_logits_to_logprobs_promotes_bf16_to_fp32():
    """Regression test for #21: log_softmax must run in fp32 even when
    the model returns bf16 logits, so per-token bf16 rounding error does
    not compound across the sequence sum.

    Starting from the same bf16-rounded logits, the fp32-internal path
    (the fix) must produce a sequence-summed log-prob closer to the full
    fp32 reference than the bf16-internal path (the unfixed code) does.
    """
    torch.manual_seed(0)
    B, L, V = 2, 256, 6  # T=256 from the issue's measurement
    fp32_logits = torch.randn(B, L, V) * 5
    bf16_logits = fp32_logits.to(torch.bfloat16)
    input_ids = torch.randint(0, V, (B, L))

    # The fix: fp32 internal even from bf16 input.
    logp_fixed = _logits_to_logprobs(bf16_logits, input_ids)
    assert logp_fixed.dtype == torch.float32

    # Full fp32 path (best attainable given fp32 logits).
    logp_fp32 = _logits_to_logprobs(fp32_logits, input_ids)

    # Unfixed path: log_softmax in bf16 (inline reproduction).
    softmax_bf16 = torch.log_softmax(bf16_logits, dim=-1)[:, :-1]
    targets = input_ids[:, 1:]
    logp_unfixed = (
        torch.gather(softmax_bf16, 2, targets.unsqueeze(-1)).squeeze(-1).float()
    )

    err_fixed = (logp_fixed.sum(-1) - logp_fp32.sum(-1)).abs().max().item()
    err_unfixed = (logp_unfixed.sum(-1) - logp_fp32.sum(-1)).abs().max().item()
    # The unfixed path compounds bf16 log_softmax error across L-1
    # positions on top of input-rounding error; the fix only carries the
    # latter.
    assert err_fixed < err_unfixed


def test_run_ll_clm_end_to_end():
    """Smoke test: run_ll_clm threads transform_ll_clm + compute_ll_clm
    through the HF Trainer batching pipeline and produces the expected
    [N, 4] shape. Catches any future regression in the wiring of the
    partial / data-transform / model-compute-fn pipeline."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)

    seqs = ["ACGTAC", "AcGtAc", "acgtac", "ACGTAC"]
    dataset = datasets.Dataset.from_dict({"seq": seqs})

    pred = run_ll_clm(
        model,
        tokenizer,
        dataset,
        data_transform_on_the_fly=True,
        inference_kwargs=dict(
            per_device_eval_batch_size=2,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
        ),
    )

    assert pred.shape == (len(seqs), 4)
    # Sanity: per-row n_upper + n_lower = L - 1 (= 6 - 1 = 5; tokenizer has no specials)
    assert (pred[:, 2] + pred[:, 3] == 5).all()
    # Row 0 and row 3 are identical sequences — same outputs.
    assert (pred[0] == pred[3]).all()
    # Row 0 (all upper) and row 2 (all lower) hit different is_upper buckets
    # but since the tokenizer is case-insensitive, the *total* sum matches.
    total_0 = pred[0, 0] + pred[0, 1]
    total_2 = pred[2, 0] + pred[2, 1]
    assert math.isclose(total_0, total_2, rel_tol=1e-5, abs_tol=1e-5)


def _write_long_fasta(tmp_path):
    """400-bp FASTA — long enough for windows up to ~200bp without N-padding."""
    fasta = ">chr1\n" + ("ACGT" * 100) + "\n"
    path = tmp_path / "long.fa"
    path.write_text(fasta)
    return path


def _make_variant_dataset():
    """A handful of variants at mid-chrom positions so windows don't N-pad.

    Ref alleles match the underlying FASTA (``"ACGT" * 100``), so for 1-based
    VCF position ``N`` the genome base is ``"ACGT"[(N - 1) % 4]``.
    """
    return datasets.Dataset.from_dict(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "pos": [99, 100, 101, 102],  # G, T, A, C
            "ref": ["G", "T", "A", "C"],
            "alt": ["C", "A", "T", "G"],
        }
    )


_INFERENCE_KWARGS = dict(
    per_device_eval_batch_size=2,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to="none",
)


def test_run_reflogprob_clm_fwd_and_rc_differ():
    """FWD and RC ``transform_reflogprob_clm`` passes through
    ``run_inference`` produce distinct outputs. ``marin_dna.model.runner``
    doesn't export a ``run_reflogprob_clm`` wrapper, so this verifies the
    underlying ``run_inference`` + ``compute_reflogprob_clm`` +
    ``transform_reflogprob_clm`` wiring at least responds to strand —
    per-strand return semantics are covered by the
    ``run_variant_score_bundle`` test below."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    model.eval()
    seqs = ["ACGTACGTACGT", "TGCATGCATGCA", "AAACCCGGGTTT", "CGATCGATCGAT"]
    pos_list = [5, 4, 6, 7]
    dataset = datasets.Dataset.from_dict({"seq": seqs, "pos": pos_list})

    fwd = run_inference(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_reflogprob_clm,
        data_transform_fn=partial(transform_reflogprob_clm, strand="+"),
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    rc = run_inference(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_reflogprob_clm,
        data_transform_fn=partial(transform_reflogprob_clm, strand="-"),
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    assert not np.allclose(fwd, rc, atol=1e-6)


class _DeterministicCausalLM(nn.Module):
    """Test double whose forward returns content-dependent logits, so that
    different input batches produce different outputs. Used to verify the
    rc=True path of ``run_variant_score_bundle`` returns distinct FWD and
    RC [N, 2] predictions and that averaging them outside the runner
    matches a manual per-strand pair.

    Mimics HF ``CausalLMOutput``: returns an object with ``.logits`` only
    (no ``.hidden_states`` — the kernel no longer requests them).

    Logits depend only on ``input_ids[t]`` at each position, not on
    surrounding context — so the prefix-shared kernel
    (``compute_variant_score_bundle``) gives bit-identical results to a
    full-sequence forward on this mock. Accepts ``use_cache`` and
    ``past_key_values`` for compatibility with the prefix-sharing call
    pattern; the cache content is dummy zeros (the mock's logits don't
    actually depend on past tokens)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        x = input_ids.float()
        logits = x.unsqueeze(-1).repeat(1, 1, self.vocab_size).contiguous()
        logits = logits + torch.arange(self.vocab_size, dtype=torch.float)
        out = SimpleNamespace(logits=logits)
        if use_cache:
            B, L = input_ids.shape
            # Minimal tuple-of-(K, V)-pairs shaped [B, num_heads=1, L, head_dim=1]
            # so `_repeat_interleave_kv_cache` exercises the legacy-format
            # branch. The mock ignores cache content; only the structure matters.
            k = torch.zeros(B, 1, L, 1)
            v = torch.zeros(B, 1, L, 1)
            out.past_key_values = ((k, v),)
        return out


def test_run_variant_score_bundle_rc_returns_both_strands(tmp_path):
    """End-to-end smoke test for the dict return shape with rc=True.

    Verifies that ``run_variant_score_bundle(rc=True)`` returns
    ``{"fwd": [N, 2], "rc": [N, 2]}`` and that the two strands agree
    with single-strand runs (callable manual two-pass reference).
    Catches regressions in the per-strand var_pos derivation,
    partial-binding, and the new dict-shape return contract."""
    from marin_dna.data.transforms import _get_special_token_counts, in_seq_var_pos

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    # songlab tokenizer puts ACGT at IDs 3-6; vocab_size=8 in the mock has
    # enough headroom for the JSD slice into the 4 nucleotide columns.
    model = _DeterministicCausalLM(vocab_size=8)
    model.eval()
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()
    window_size = 16  # even → FWD and RC have different in-seq var_pos

    n_prefix, _ = _get_special_token_counts(tokenizer)
    nuc_ids_dict = _get_nucleotide_token_ids(tokenizer)
    nuc_token_ids = torch.tensor(
        [nuc_ids_dict[nuc] for nuc in NUCLEOTIDES], dtype=torch.long
    )
    var_pos_rc = in_seq_var_pos(window_size, "-") + n_prefix

    fwd_only = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=False,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    rc_manual = run_inference(
        model,
        tokenizer,
        dataset,
        compute_fn=partial(
            compute_variant_score_bundle,
            var_pos=var_pos_rc,
            nuc_token_ids=nuc_token_ids,
        ),
        data_transform_fn=partial(
            transform_llr_clm, genome=genome, window_size=window_size, strand="-"
        ),
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    both = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )

    assert set(fwd_only.keys()) == {"fwd"}
    assert set(both.keys()) == {"fwd", "rc"}
    assert fwd_only["fwd"].shape == (4, 2)
    assert both["fwd"].shape == (4, 2)
    assert both["rc"].shape == (4, 2)
    np.testing.assert_allclose(both["fwd"], fwd_only["fwd"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(both["rc"], rc_manual, rtol=1e-5, atol=1e-6)
    assert not np.allclose(both["fwd"], both["rc"], atol=1e-6)


def test_run_variant_score_bundle_rc_bitwise_reproducible(tmp_path):
    """Two runs of ``run_variant_score_bundle(rc=True)`` on identical
    inputs must produce bit-identical dicts. snakemake's `params:` rerun
    trigger compares output hashes — a non-determinism regression here
    would silently break the "re-run on revision bump" contract."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = _DeterministicCausalLM(vocab_size=8)
    model.eval()
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()
    window_size = 16

    out_a = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    out_b = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    assert set(out_a.keys()) == set(out_b.keys()) == {"fwd", "rc"}
    np.testing.assert_array_equal(out_a["fwd"], out_b["fwd"])
    np.testing.assert_array_equal(out_a["rc"], out_b["rc"])


def test_run_inference_padding_roundtrip(tmp_path):
    """When n_examples is not a multiple of batch_size, ``_run_inference``
    pads the dataset to a clean multiple (so torch.compile sees only one
    batch shape) and slices the padded predictions off before returning.

    Set up a dataset of 4 variants with batch_size=3 → pads by 2 internally.
    Verify (a) shape is (4, 2) — padding is invisible to the caller — and
    (b) the per-row predictions match what we get with batch_size=2 (no pad)."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = _DeterministicCausalLM(vocab_size=8)
    model.eval()
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()
    window_size = 16

    no_pad = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=False,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,  # batch_size=2 → divides 4 evenly
    )

    padded_kwargs = {**_INFERENCE_KWARGS, "per_device_eval_batch_size": 3}
    with_pad = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=False,
        data_transform_on_the_fly=True,
        inference_kwargs=padded_kwargs,  # batch_size=3 → pads 4→6, slices 6→4
    )

    assert no_pad["fwd"].shape == (4, 2)
    assert with_pad["fwd"].shape == (4, 2)
    np.testing.assert_allclose(no_pad["fwd"], with_pad["fwd"], rtol=1e-5, atol=1e-6)


class _ContentIndependentCausalLM(nn.Module):
    """Test double whose forward returns logits independent of input_ids
    content (only depends on absolute position and vocab indices). Used to
    verify that the JSD column is exactly 0 when ref and alt sequences
    produce identical next-token distributions.

    Position-aware: derives the absolute-position offset from the length
    of any provided ``past_key_values``. So the prefix-shared call pattern
    (prefix forward at offset 0, suffix forward at offset = prefix length)
    yields the same per-position logits as a single full-sequence forward.
    This is what catches a bug where ``compute_variant_score_bundle``
    forgot to pass ``past_key_values`` to the suffix call — the suffix
    would see offset 0 instead of ``var_pos``, mismatching the full-forward
    reference."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        B, L = input_ids.shape
        offset = _kv_cache_seq_len(past_key_values)
        pos = torch.arange(offset, offset + L, dtype=torch.float).unsqueeze(-1)
        vocab = torch.arange(self.vocab_size, dtype=torch.float)
        logits_per_seq = pos + vocab  # [L, V]
        logits = logits_per_seq.unsqueeze(0).expand(B, L, self.vocab_size).contiguous()
        out = SimpleNamespace(logits=logits)
        if use_cache:
            k = torch.zeros(B, 1, L, 1)
            v = torch.zeros(B, 1, L, 1)
            out.past_key_values = ((k, v),)
        return out


def _kv_cache_seq_len(past_kv: object) -> int:
    """Length of the cached prefix (0 if absent).

    Handles both the legacy tuple-of-(K, V) cache and HF's ``DynamicCache``:
    production coerces this mock's legacy tuple into a ``DynamicCache``
    (``scoring._repeat_interleave_kv_cache``), and transformers >= 5 dropped
    tuple-style subscripting on ``Cache`` objects, so read the length via the
    public API when it's present. K shape is [B, num_heads, seq_len, head_dim];
    dim 2 is seq_len."""
    if past_kv is None:
        return 0
    if hasattr(past_kv, "get_seq_length"):
        return int(past_kv.get_seq_length())
    return int(past_kv[0][0].shape[2])


def test_next_token_jsd_mean_zero_when_ref_alt_logits_identical(tmp_path):
    """If logits don't depend on input_ids content, the per-position 4-nuc
    softmax is identical for ref and alt → JSD = 0 at every position →
    next_token_jsd_mean = 0 for every batch row."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = _ContentIndependentCausalLM(vocab_size=8)
    model.eval()
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()
    window_size = 16

    out = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=False,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    assert out["fwd"].shape == (4, 2)
    np.testing.assert_allclose(out["fwd"][:, 1], 0.0, atol=1e-7)


def test_compute_variant_score_bundle_prefix_sharing_correctness():
    """Prefix-shared kernel produces correct LLR + zero JSD on a position-aware
    mock whose logits depend on absolute position only (not on input content).

    This catches two classes of bug:

    1. **Forgot to pass past_key_values to the suffix forward**: the suffix
       would see absolute positions [0, L-p) instead of [p, L), shifting the
       logits and breaking LLR_at_var.
    2. **JSD computation includes positions where ref/alt distributions
       differ**: for this mock all positions give identical distributions
       (content-independent), so JSD must be exactly 0 at every position.

    Constructs LLR analytically from the mock's logit formula: at the prefix's
    last position (var_pos - 1), 4-nuc logits are
    ``[var_pos-1+nuc_id for nuc_id in [3,4,5,6]]`` (offset = past_kv_len = 0
    for the prefix forward). After log_softmax, gather at alt-nuc-idx vs
    ref-nuc-idx and difference."""
    torch.manual_seed(0)
    model = _ContentIndependentCausalLM(vocab_size=8)
    model.eval()
    nuc_token_ids = torch.tensor([3, 4, 5, 6], dtype=torch.long)

    L = 10
    var_pos = 4
    # Ref rows: arbitrary; alt token differs from ref at var_pos.
    # Row 0: ref nuc at var_pos = 4 (idx 1 in nuc_ids); alt = 5 (idx 2).
    # Row 1: ref nuc at var_pos = 5 (idx 2);             alt = 3 (idx 0).
    input_ids = torch.tensor(
        [
            [3, 4, 5, 3, 4, 5, 6, 3, 4, 5],
            [4, 5, 6, 4, 5, 6, 3, 4, 5, 6],
        ]
    )
    alt_token_id = torch.tensor([5, 3])
    assert input_ids.shape == (2, L)

    out = compute_variant_score_bundle(
        model,
        input_ids,
        alt_token_id,
        var_pos=var_pos,
        nuc_token_ids=nuc_token_ids,
    )
    assert out.shape == (2, 2)

    # Expected LLR_at_var: log_p[alt_idx] - log_p[ref_idx], where log_p
    # is log_softmax of the mock's 4-nuc logits at position var_pos - 1.
    # Mock logits[t, v] = t + v; offset from prefix is 0 (no past_kv).
    log_p = torch.log_softmax(
        torch.tensor([(var_pos - 1) + v for v in [3, 4, 5, 6]], dtype=torch.float),
        dim=-1,
    )
    expected_llr = torch.tensor(
        [
            log_p[2] - log_p[1],  # row 0: alt=5(idx2), ref=4(idx1)
            log_p[0] - log_p[2],  # row 1: alt=3(idx0), ref=5(idx2)
        ]
    )
    np.testing.assert_allclose(out[:, 0].numpy(), expected_llr.numpy(), atol=1e-5)

    # JSD = 0: content-independent logits → identical ref/alt distributions
    # at every position → KL(P||M) = KL(Q||M) = 0.
    np.testing.assert_allclose(out[:, 1].numpy(), 0.0, atol=1e-7)


def test_compute_variant_score_bundle_jsd_analytic():
    """Hand-craft logits where the per-position 4-nuc JSD is computable
    analytically and assert numerical match.

    Setup: B=1, L=4, V=4 (= nuc_token_ids = [0, 1, 2, 3] so the 4-nuc slice
    is the identity). var_pos=1. Ref input_ids = [0, 0, 0, 0]; alt token at
    var_pos = 1 (so reconstructed alt = [0, 1, 0, 0]).

    Mock logits: ``[10, 0, 0, 0]`` (sharp toward token 0) where input_ids[t]
    is nonzero, else uniform ``[0, 0, 0, 0]``. Two downstream positions
    (t in [1, 2] = suffix indices [0, 1]):

    - Suffix-pos 0 (= global pos 1): ref-suffix[0]=0 → uniform; alt-suffix[0]=1
      → sharp. JSD nonzero (uniform vs sharp).
    - Suffix-pos 1 (= global pos 2): both ref-suffix[1]=alt-suffix[1]=0 →
      uniform. JSD = 0.

    Mean JSD = jsd_at_var / 2."""

    class _Custom(nn.Module):
        def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
            B, L = input_ids.shape
            V = 4
            logits = torch.zeros(B, L, V)
            # Sharp toward token 0 where input_ids != 0; else uniform zeros.
            sharp = (input_ids != 0).float().unsqueeze(-1)
            template = torch.tensor([10.0, 0.0, 0.0, 0.0])
            logits = logits + sharp * template
            out = SimpleNamespace(logits=logits)
            if use_cache:
                k = torch.zeros(B, 1, L, 1)
                v = torch.zeros(B, 1, L, 1)
                out.past_key_values = ((k, v),)
            return out

    model = _Custom()
    model.eval()
    input_ids = torch.tensor([[0, 0, 0, 0]])  # [B=1, L=4] — ref only
    alt_token_id = torch.tensor([1])  # alt nuc at var_pos
    nuc_token_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    out = compute_variant_score_bundle(
        model, input_ids, alt_token_id, var_pos=1, nuc_token_ids=nuc_token_ids
    )
    assert out.shape == (1, 2)

    # Analytic JSD at suffix position 0 (global pos 1):
    #   ref distribution = log_softmax([0,0,0,0]) = uniform
    #   alt distribution = log_softmax([10,0,0,0]) = sharp toward token 0
    p_ref = torch.full((4,), 0.25)
    log_p_ref = p_ref.log()
    log_p_alt = torch.log_softmax(torch.tensor([10.0, 0.0, 0.0, 0.0]), dim=-1)
    p_alt = log_p_alt.exp()
    log_m = torch.logaddexp(log_p_ref, log_p_alt) - math.log(2.0)
    kl_ref_m = (p_ref * (log_p_ref - log_m)).sum()
    kl_alt_m = (p_alt * (log_p_alt - log_m)).sum()
    jsd_at_var = 0.5 * (kl_ref_m + kl_alt_m).item()

    # 2 downstream positions, JSD nonzero only at the first.
    expected_mean = jsd_at_var / 2
    np.testing.assert_allclose(out[0, 1].item(), expected_mean, rtol=1e-5)
