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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

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
    run_variant_marginal,
    run_variant_score_bundle,
)
from marin_dna.model.scoring import (
    _logits_to_logprobs,
    _token_id_to_nuc_idx,
    compute_ll_clm,
    compute_marginal_clm,
    compute_reflogprob_clm,
    compute_variant_llr,
    compute_variant_llr_branch_packed,
    compute_variant_llr_full_pair,
    compute_variant_llr_sequential_branches,
    compute_variant_score_bundle,
    entropy_from_marginal,
    make_variant_branch_packed_layout,
    rc_average_marginal,
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


def test_compute_variant_llr_matches_bundle_exactly():
    """The LLR-only kernel preserves the existing bundle's score contract."""
    torch.manual_seed(0)
    model = _DeterministicCausalLM(vocab_size=8)
    model.eval()
    input_ids = torch.randint(3, 7, (5, 18))
    alt_token_id = torch.tensor([3, 4, 5, 6, 3])
    var_pos = 8

    bundled = compute_variant_score_bundle(
        model,
        input_ids,
        alt_token_id,
        var_pos=var_pos,
        nuc_token_ids=_SONGLAB_NUC_TOKEN_IDS,
    )
    llr_only = compute_variant_llr(
        model,
        input_ids,
        alt_token_id,
        var_pos=var_pos,
        nuc_token_ids=_SONGLAB_NUC_TOKEN_IDS,
    )
    assert llr_only.shape == (len(input_ids),)
    torch.testing.assert_close(llr_only, bundled[:, 0], rtol=0, atol=0)


def test_make_variant_branch_packed_layout_is_two_isolated_causal_branches():
    position_ids, attention_mask = make_variant_branch_packed_layout(
        sequence_length=6,
        var_pos=3,
    )
    assert position_ids.tolist() == [[0, 1, 2, 3, 4, 5, 3, 4, 5]]
    allowed = attention_mask[0, 0]
    expected_keys = {
        0: [0],
        1: [0, 1],
        2: [0, 1, 2],
        3: [0, 1, 2, 3],
        4: [0, 1, 2, 3, 4],
        5: [0, 1, 2, 3, 4, 5],
        6: [0, 1, 2, 6],
        7: [0, 1, 2, 6, 7],
        8: [0, 1, 2, 6, 7, 8],
    }
    for query, keys in expected_keys.items():
        observed = torch.nonzero(allowed[query], as_tuple=False).flatten().tolist()
        assert observed == keys


def test_variant_llr_cache_branch_packed_and_full_pair_agree_on_qwen3():
    """All exact execution layouts preserve Qwen3 LLRs up to fp32 noise."""
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=7,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=32,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=None,
        pad_token_id=None,
    )
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.randint(3, 7, (3, 12))
    input_ids[:, 0] = 2
    var_pos = 6
    ref_token_id = input_ids[:, var_pos]
    alt_token_id = 3 + ((ref_token_id - 3 + 1) % 4)
    nuc_token_ids = torch.tensor([3, 4, 5, 6])

    with torch.inference_mode():
        cached = compute_variant_llr(
            model,
            input_ids,
            alt_token_id,
            var_pos=var_pos,
            nuc_token_ids=nuc_token_ids,
        )
        sequential = compute_variant_llr_sequential_branches(
            model,
            input_ids,
            alt_token_id,
            var_pos=var_pos,
            nuc_token_ids=nuc_token_ids,
        )
        branch_packed = compute_variant_llr_branch_packed(
            model,
            input_ids,
            alt_token_id,
            var_pos=var_pos,
            nuc_token_ids=nuc_token_ids,
        )
        full_pair = compute_variant_llr_full_pair(
            model,
            input_ids,
            alt_token_id,
            var_pos=var_pos,
            nuc_token_ids=nuc_token_ids,
        )

    torch.testing.assert_close(branch_packed, cached, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(full_pair, cached, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sequential, cached, rtol=1e-5, atol=1e-5)


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


# --- compute_marginal_clm / entropy_from_marginal --------------------------

_SONGLAB_NUC_TOKEN_IDS = torch.tensor([3, 4, 5, 6], dtype=torch.long)  # A C G T


def test_compute_marginal_clm_is_log_prob_distribution():
    """Each row of the [B, 4] marginal is a normalized log-prob (logsumexp == 0)."""
    torch.manual_seed(0)
    model = _DeterministicCausalLM(vocab_size=8)
    model.eval()
    B, L, var_pos = 3, 12, 5
    # Nucleotide-only token ids (songlab ACGT live at 3-6).
    input_ids = torch.randint(3, 7, (B, L))
    out = compute_marginal_clm(
        model, input_ids, var_pos=var_pos, nuc_token_ids=_SONGLAB_NUC_TOKEN_IDS
    )
    assert out.shape == (B, 4)
    np.testing.assert_allclose(
        torch.logsumexp(out, dim=-1).detach().numpy(), 0.0, atol=1e-5
    )
    # log of a probability ⇒ every entry ≤ 0.
    assert (out <= 1e-6).all()


def test_compute_marginal_clm_matches_bundle_llr():
    """``marginal[:, alt] − marginal[:, ref]`` is identically
    ``compute_variant_score_bundle``'s LLR.

    Both kernels build the same ref/alt suffixes against the same shared prefix
    and reduce in the same 4-nuc-softmax space, so the allele difference of the
    marginal equals the bundle's LLR column to fp precision for *every* alt.
    This is the consistency contract that lets calibration LLRs reuse the eval
    LLR definition for free."""
    torch.manual_seed(0)
    model = _DeterministicCausalLM(vocab_size=8)
    model.eval()
    B, L, var_pos = 5, 14, 6
    input_ids = torch.randint(3, 7, (B, L))

    marginal = compute_marginal_clm(
        model, input_ids, var_pos=var_pos, nuc_token_ids=_SONGLAB_NUC_TOKEN_IDS
    )  # [B, 4]
    ref_idx = _token_id_to_nuc_idx(input_ids[:, var_pos], _SONGLAB_NUC_TOKEN_IDS)
    rows = torch.arange(B)
    for alt_nuc_idx in range(4):
        alt_token_id = _SONGLAB_NUC_TOKEN_IDS[alt_nuc_idx].repeat(B)
        bundle = compute_variant_score_bundle(
            model,
            input_ids,
            alt_token_id,
            var_pos=var_pos,
            nuc_token_ids=_SONGLAB_NUC_TOKEN_IDS,
        )
        marginal_llr = marginal[rows, alt_nuc_idx] - marginal[rows, ref_idx]
        np.testing.assert_allclose(
            marginal_llr.detach().numpy(),
            bundle[:, 0].detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
        )


def test_entropy_from_marginal_range_and_endpoints():
    """Entropy ∈ [0, log 4]; log 4 at uniform, → 0 at a near-degenerate marginal."""
    uniform = np.log(np.full((3, 4), 0.25))
    np.testing.assert_allclose(entropy_from_marginal(uniform), math.log(4), atol=1e-6)
    degen = np.log(np.array([[0.9997, 0.0001, 0.0001, 0.0001]]))
    ent_degen = entropy_from_marginal(degen)
    assert 0.0 <= ent_degen[0] < 0.01
    # Random valid marginals stay in range.
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((20, 4))
    marg = logits - np.log(np.exp(logits).sum(-1, keepdims=True))  # log_softmax
    ent = entropy_from_marginal(marg)
    assert (ent >= -1e-9).all() and (ent <= math.log(4) + 1e-9).all()


def test_entropy_from_marginal_matches_reduction_on_compute_marginal():
    """``entropy_from_marginal`` equals ``−(m.exp()·m).sum()`` of
    ``compute_marginal_clm``'s output."""
    torch.manual_seed(0)
    model = _DeterministicCausalLM(vocab_size=8)
    model.eval()
    input_ids = torch.randint(3, 7, (4, 12))
    m = compute_marginal_clm(
        model, input_ids, var_pos=5, nuc_token_ids=_SONGLAB_NUC_TOKEN_IDS
    )
    expected = -(m.exp() * m).sum(dim=-1)
    np.testing.assert_allclose(
        entropy_from_marginal(m.detach().numpy()),
        expected.detach().numpy(),
        atol=1e-5,
    )


def test_entropy_from_marginal_permutation_invariant():
    """Permuting the 4 allele columns (the A↔T / C↔G complement relabel between
    FWD and RC) leaves entropy unchanged."""
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((10, 4))
    marg = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    comp = marg[:, [3, 2, 1, 0]]  # A↔T (0↔3), C↔G (1↔2)
    np.testing.assert_allclose(
        entropy_from_marginal(marg), entropy_from_marginal(comp), atol=1e-12
    )


def test_entropy_from_marginal_normalizes_internally():
    """``entropy_from_marginal`` accepts unnormalized logits (it normalizes
    internally) — so a plain-mean FWD+RC average of log-probs needs no separate
    renormalization. Per-row additive shifts leave the entropy unchanged."""
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((8, 4)) * 3.0  # arbitrary, unnormalized
    norm = logits - np.log(np.exp(logits).sum(-1, keepdims=True))  # log_softmax
    expected = -(np.exp(norm) * norm).sum(-1)
    np.testing.assert_allclose(entropy_from_marginal(logits), expected, atol=1e-10)
    # softmax is shift-invariant → entropy of (logits + per-row const) is unchanged.
    shifted = logits + rng.standard_normal((8, 1)) * 10.0
    np.testing.assert_allclose(
        entropy_from_marginal(logits), entropy_from_marginal(shifted), atol=1e-10
    )


def test_rc_average_marginal_realigns_complement_columns():
    """``rc_average_marginal`` realigns the RC complement columns before averaging.

    If the RC marginal labels the *same* distribution as FWD but by complement
    (columns permuted ``[3,2,1,0]``), the aligned average recovers FWD exactly;
    a naive ``(fwd + rc)/2`` would not."""
    rng = np.random.default_rng(4)
    fwd = rng.standard_normal((5, 4))
    rc_same_dist = fwd[:, [3, 2, 1, 0]]  # RC names the same distribution by complement
    np.testing.assert_allclose(rc_average_marginal(fwd, rc_same_dist), fwd, atol=1e-12)
    assert not np.allclose(0.5 * (fwd + rc_same_dist), fwd, atol=1e-3)


def test_rc_average_marginal_llr_matches_bundle_rc_average(tmp_path):
    """End-to-end: the LLR read off ``rc_average_marginal`` equals the eval
    bundle's FWD+RC-averaged LLR — validating both the complement realignment and
    the average-then-derive ≡ derive-then-average linearity."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    model.eval()
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()
    window_size = 16

    marg = run_variant_marginal(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    avg = rc_average_marginal(marg["fwd"], marg["rc"])  # [N, 4], forward ACGT order

    bundle = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    bundle_llr_avg = 0.5 * (bundle["fwd"][:, 0] + bundle["rc"][:, 0])

    ref_idx = np.array([NUCLEOTIDES.index(r) for r in dataset["ref"]])
    alt_idx = np.array([NUCLEOTIDES.index(a) for a in dataset["alt"]])
    rows = np.arange(len(dataset))
    marg_avg_llr = avg[rows, alt_idx] - avg[rows, ref_idx]
    np.testing.assert_allclose(marg_avg_llr, bundle_llr_avg, rtol=1e-4, atol=1e-4)


def test_run_variant_marginal_rc_returns_both_strands(tmp_path):
    """``run_variant_marginal(rc=True)`` returns ``{"fwd": [N,4], "rc": [N,4]}``
    with distinct, normalized per-strand marginals; ``rc=False`` returns only
    ``fwd`` and agrees with the rc=True fwd.

    Uses the real tiny CLM rather than the content-independent mock: the mock's
    per-position logits depend only on that position's token, so the variant-site
    marginal collapses to a constant and FWD/RC would be indistinguishable. The
    real model's context dependence makes the two strands genuinely differ."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    model.eval()
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()
    window_size = 16

    both = run_variant_marginal(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    fwd_only = run_variant_marginal(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=False,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    assert set(fwd_only.keys()) == {"fwd"}
    assert set(both.keys()) == {"fwd", "rc"}
    assert both["fwd"].shape == (4, 4)
    assert both["rc"].shape == (4, 4)
    np.testing.assert_allclose(both["fwd"], fwd_only["fwd"], rtol=1e-5, atol=1e-6)
    # Each row is a normalized log-prob marginal.
    np.testing.assert_allclose(
        np.logaddexp.reduce(both["fwd"], axis=-1), 0.0, atol=1e-5
    )
    assert not np.allclose(both["fwd"], both["rc"], atol=1e-6)


def test_run_variant_marginal_reproducible(tmp_path):
    """Two identical runs produce bit-identical marginals (snakemake `params:`
    rerun-trigger contract — see the bundle's reproducibility test)."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = _DeterministicCausalLM(vocab_size=8)
    model.eval()
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()

    a = run_variant_marginal(
        model,
        tokenizer,
        dataset,
        genome,
        16,
        rc=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    b = run_variant_marginal(
        model,
        tokenizer,
        dataset,
        genome,
        16,
        rc=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    np.testing.assert_array_equal(a["fwd"], b["fwd"])
    np.testing.assert_array_equal(a["rc"], b["rc"])


def test_run_variant_marginal_matches_bundle_llr_end_to_end(tmp_path):
    """End-to-end: ``marginal[:, alt] − marginal[:, ref]`` (FWD) matches
    ``run_variant_score_bundle``'s FWD LLR on the same variants — the transform
    + kernel + runner wiring agrees with the production LLR path.

    On the real tiny CLM (genuine context dependence) both prefix-shared kernels
    see the same cached prefix and the same ref/alt suffixes, so their LLRs agree
    to fp precision."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    model.eval()
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()
    window_size = 16

    marg = run_variant_marginal(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=False,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    bundle = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=False,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    # FWD strand: marginal column index == NUCLEOTIDES.index(letter).
    ref_idx = np.array([NUCLEOTIDES.index(r) for r in dataset["ref"]])
    alt_idx = np.array([NUCLEOTIDES.index(a) for a in dataset["alt"]])
    rows = np.arange(len(dataset))
    marg_llr = marg["fwd"][rows, alt_idx] - marg["fwd"][rows, ref_idx]
    np.testing.assert_allclose(marg_llr, bundle["fwd"][:, 0], rtol=1e-4, atol=1e-4)


# --- compute_variant_score_bundle return_embeddings (issue #318) -------------


class _FixedHiddenBase(nn.Module):
    """Stand-in for an HF base/decoder model: returns the fixed per-position
    hidden as ``last_hidden_state``, so a forward hook on it captures the last
    layer — exactly how ``compute_variant_score_bundle`` grabs hidden states
    without ``output_hidden_states``. Position-aware via the cached-prefix offset
    (``_kv_cache_seq_len``); ``logits_to_keep`` must NOT truncate it (mirrors HF),
    so it always returns the full ``[B, L, D]`` window."""

    def __init__(self, hidden_full: Tensor):
        super().__init__()
        self.register_buffer("hidden_full", hidden_full)  # [L_max, D]

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        B, L = input_ids.shape
        offset = _kv_cache_seq_len(past_key_values)
        h = self.hidden_full[offset : offset + L]  # [L, D]
        h = h.unsqueeze(0).expand(B, L, h.shape[-1]).contiguous()
        pkv = None
        if use_cache:
            pkv = ((torch.zeros(B, 1, L, 1), torch.zeros(B, 1, L, 1)),)
        return SimpleNamespace(last_hidden_state=h, past_key_values=pkv)


class _FixedHiddenCausalLM(nn.Module):
    """Test double exposing a *fixed* per-position last-layer hidden state, so the
    pooled embedding has a closed form.

    ``hidden_full`` is ``[L_max, D]``: the last-layer state at absolute position
    ``t`` is ``hidden_full[t]``, independent of token content. The ``base_model``
    submodule returns it as ``last_hidden_state``, so the kernel's forward hook on
    ``model.base_model`` captures the matching slice for the prefix-shared call
    pattern (prefix forward at offset 0, suffix forward at offset = prefix length)
    — exactly what ``compute_variant_score_bundle`` concatenates. Logits are
    content-independent (``pos + vocab``) — valid for the 4-nuc LLR/JSD path but
    irrelevant to the embedding columns under test. Content-independence ⇒
    ``emb_ref == emb_alt`` (the only ref/alt difference is the token at
    ``var_pos``, which doesn't move the hidden state here)."""

    def __init__(self, hidden_full: Tensor, vocab_size: int):
        super().__init__()
        self.base_model = _FixedHiddenBase(hidden_full)
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        **kwargs,
    ):
        # Route through base_model so a forward hook on it fires (the kernel's
        # no-all-layers hidden-state capture path).
        base = self.base_model(
            input_ids, past_key_values=past_key_values, use_cache=use_cache
        )
        B, L = input_ids.shape
        offset = _kv_cache_seq_len(past_key_values)
        pos = torch.arange(offset, offset + L, dtype=torch.float).unsqueeze(-1)
        vocab = torch.arange(self.vocab_size, dtype=torch.float)
        logits = (pos + vocab).unsqueeze(0).expand(B, L, self.vocab_size).contiguous()
        out = SimpleNamespace(logits=logits)
        if use_cache:
            out.past_key_values = base.past_key_values
        return out


_IDENTITY_NUC_IDS = torch.tensor([0, 1, 2, 3], dtype=torch.long)


def test_variant_score_bundle_embeddings_pool_math_and_bos_exclusion():
    """Entire-window pool is the fp32 mean over [pool_lo, pool_hi) DNA positions,
    with the n_prefix BOS token(s) excluded.

    Fixed hidden state ``h(t)[d] = t + d`` ⇒ the pooled embedding is
    ``mean_{t in [lo,hi)}(t) + d`` in closed form. ``pool_lo=1`` drops position 0
    (the BOS analog); ``pool_hi=L-1`` drops the trailing position — so the pool is
    exactly the inner DNA block, and including position 0 would shift the mean."""
    L, D, vocab = 12, 5, 8
    var_pos = 4
    # h(t)[d] = t + d
    hidden_full = (
        torch.arange(L, dtype=torch.float)[:, None]
        + torch.arange(D, dtype=torch.float)[None, :]
    )  # [L, D]
    model = _FixedHiddenCausalLM(hidden_full, vocab_size=vocab)
    model.eval()
    input_ids = torch.randint(0, 4, (3, L))  # nucleotide tokens (ids 0-3)
    alt_token_id = torch.randint(0, 4, (3,))
    pool_lo, pool_hi = 1, L - 1  # exclude BOS (idx 0) and trailing (idx L-1)

    out = compute_variant_score_bundle(
        model,
        input_ids,
        alt_token_id,
        var_pos=var_pos,
        nuc_token_ids=_IDENTITY_NUC_IDS,
        return_embeddings=True,
        pool_lo=pool_lo,
        pool_hi=pool_hi,
    )
    assert out.shape == (3, 2 + 2 * D)
    emb_ref = out[:, 2 : 2 + D].numpy()
    emb_alt = out[:, 2 + D : 2 + 2 * D].numpy()
    # Content-independent hidden ⇒ ref/alt pools identical.
    np.testing.assert_allclose(emb_ref, emb_alt, atol=1e-5)
    # Closed form: mean over t in [pool_lo, pool_hi) of (t + d).
    mean_t = float(np.mean(np.arange(pool_lo, pool_hi)))
    expected = mean_t + np.arange(D)
    np.testing.assert_allclose(emb_ref[0], expected, atol=1e-4)
    # Pooled count == window_size, BOS excluded: pooling [0, pool_hi) would shift
    # the mean down — the shipped pool must NOT equal that.
    mean_with_bos = float(np.mean(np.arange(0, pool_hi)))
    assert not np.allclose(emb_ref[0], mean_with_bos + np.arange(D), atol=1e-3)
    assert np.isfinite(out.numpy()).all()


def test_variant_score_bundle_embeddings_accumulate_in_fp32():
    """The window pool accumulates in fp32, not the model's (bf16) hidden dtype.

    With bf16 hidden states summed over many positions, the kernel's fp32-cast
    pool matches an fp32 accumulation of those bf16 values and DIFFERS measurably
    from a bf16 accumulation — the latter is the silent-corruption path the
    ``.float()`` cast prevents (cf. ``test_logits_to_logprobs_promotes_bf16_to_fp32``)."""
    torch.manual_seed(0)
    L, D, vocab = 20, 6, 8
    var_pos = 7
    pool_lo, pool_hi = 1, L - 1
    hidden_full = (torch.randn(L, D) * 50.0).to(torch.bfloat16)  # large ⇒ bf16 rounds
    model = _FixedHiddenCausalLM(hidden_full, vocab_size=vocab)
    model.eval()
    input_ids = torch.randint(0, 4, (2, L))
    alt_token_id = torch.randint(0, 4, (2,))

    out = compute_variant_score_bundle(
        model,
        input_ids,
        alt_token_id,
        var_pos=var_pos,
        nuc_token_ids=_IDENTITY_NUC_IDS,
        return_embeddings=True,
        pool_lo=pool_lo,
        pool_hi=pool_hi,
    )
    emb_kernel = out[:, 2 : 2 + D]
    assert emb_kernel.dtype == torch.float32, "pooled embedding must be fp32"

    window = hidden_full[pool_lo:pool_hi]  # [n_pool, D] bf16
    emb_fp32 = window.float().mean(0)  # fp32 accumulation (what the kernel does)
    emb_bf16 = window.mean(0).float()  # bf16 accumulation (the wrong way)
    # Kernel matches the fp32 accumulation (rows identical — content-independent).
    np.testing.assert_allclose(
        emb_kernel.numpy(), emb_fp32.numpy()[None].repeat(2, 0), atol=1e-2
    )
    # ...and the fp32 vs bf16 accumulations differ by a meaningful margin.
    assert (emb_fp32 - emb_bf16).abs().max().item() > 0.05


def test_variant_score_bundle_embeddings_match_full_forward_pool():
    """On a real CLM, the prefix-shared pooled embeddings equal a naive
    full-forward-then-mean-pool over the same window — the KV-cache concat
    reconstructs the full-window hidden states, and ref/alt genuinely differ."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    model.eval()
    D = model.config.hidden_size
    L, var_pos = 12, 5
    pool_lo, pool_hi = 1, L - 1
    # Nucleotide tokens (songlab ACGT live at ids 3-6). Force alt != ref at var_pos.
    input_ids = torch.randint(3, 7, (3, L))
    alt_token_id = ((input_ids[:, var_pos] - 3 + 1) % 4) + 3

    with torch.no_grad():
        out = compute_variant_score_bundle(
            model,
            input_ids,
            alt_token_id,
            var_pos=var_pos,
            nuc_token_ids=_SONGLAB_NUC_TOKEN_IDS,
            return_embeddings=True,
            pool_lo=pool_lo,
            pool_hi=pool_hi,
        )
        ref_ids = input_ids
        alt_ids = input_ids.clone()
        alt_ids[:, var_pos] = alt_token_id
        ref_h = model(ref_ids, output_hidden_states=True).hidden_states[-1]
        alt_h = model(alt_ids, output_hidden_states=True).hidden_states[-1]
        exp_ref = ref_h[:, pool_lo:pool_hi].float().mean(1)
        exp_alt = alt_h[:, pool_lo:pool_hi].float().mean(1)

    assert out.shape == (3, 2 + 2 * D)
    np.testing.assert_allclose(out[:, 2 : 2 + D].numpy(), exp_ref.numpy(), atol=1e-4)
    np.testing.assert_allclose(
        out[:, 2 + D : 2 + 2 * D].numpy(), exp_alt.numpy(), atol=1e-4
    )
    # ref and alt embeddings genuinely differ (context-dependent model).
    assert not np.allclose(
        out[:, 2 : 2 + D].numpy(), out[:, 2 + D : 2 + 2 * D].numpy(), atol=1e-4
    )


def test_variant_score_bundle_scores_unchanged_by_embeddings():
    """The LLR/JSD columns are bit-identical with embeddings on vs off — the same
    two forwards, with ``output_hidden_states`` only *adding* outputs."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    model.eval()
    L, var_pos = 12, 5
    input_ids = torch.randint(3, 7, (4, L))
    alt_token_id = ((input_ids[:, var_pos] - 3 + 2) % 4) + 3

    with torch.no_grad():
        off = compute_variant_score_bundle(
            model,
            input_ids,
            alt_token_id,
            var_pos=var_pos,
            nuc_token_ids=_SONGLAB_NUC_TOKEN_IDS,
        )
        on = compute_variant_score_bundle(
            model,
            input_ids,
            alt_token_id,
            var_pos=var_pos,
            nuc_token_ids=_SONGLAB_NUC_TOKEN_IDS,
            return_embeddings=True,
            pool_lo=1,
            pool_hi=L - 1,
        )
    assert off.shape == (4, 2)
    np.testing.assert_array_equal(off.numpy(), on[:, :2].numpy())


def test_variant_score_bundle_embeddings_require_pool_bounds():
    """``return_embeddings=True`` without pool bounds is a loud error."""
    import pytest

    model = _FixedHiddenCausalLM(torch.zeros(8, 3), vocab_size=8)
    model.eval()
    input_ids = torch.randint(0, 4, (1, 8))
    alt_token_id = torch.randint(0, 4, (1,))
    with pytest.raises(AssertionError):
        compute_variant_score_bundle(
            model,
            input_ids,
            alt_token_id,
            var_pos=3,
            nuc_token_ids=_IDENTITY_NUC_IDS,
            return_embeddings=True,
        )


def test_variant_score_bundle_embeddings_hook_removed_and_off_path_clean():
    """The base-model forward hook is registered only when pooling and removed
    after the call (no leak across the per-batch kernel calls Trainer.predict
    makes); the off-path registers none at all."""
    D = 4
    model = _FixedHiddenCausalLM(
        torch.arange(10 * D, dtype=torch.float).reshape(10, D), vocab_size=8
    )
    model.eval()
    input_ids = torch.randint(0, 4, (2, 10))
    alt_token_id = torch.randint(0, 4, (2,))
    kw = dict(var_pos=4, nuc_token_ids=_IDENTITY_NUC_IDS)

    compute_variant_score_bundle(
        model,
        input_ids,
        alt_token_id,
        return_embeddings=True,
        pool_lo=1,
        pool_hi=9,
        **kw,
    )
    assert not model.base_model._forward_hooks, "forward hook leaked after the call"

    compute_variant_score_bundle(model, input_ids, alt_token_id, **kw)  # off-path
    assert not model.base_model._forward_hooks


def test_run_variant_score_bundle_return_embeddings_end_to_end(tmp_path):
    """``run_variant_score_bundle(return_embeddings=True)`` threads the wide
    ``[B, 2+2D]`` tensor through the HF Trainer harness (transform + batching +
    pad/slice) and binds the strand-independent pool bounds. The ``[:, :2]``
    scores match the embeddings-off run; the embedding block is ``2*D`` wide and
    finite, on both strands."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    model.eval()
    D = model.config.hidden_size
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()
    window_size = 16

    scores_only = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    with_emb = run_variant_score_bundle(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc=True,
        return_embeddings=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    for strand in ("fwd", "rc"):
        assert with_emb[strand].shape == (4, 2 + 2 * D)
        np.testing.assert_allclose(
            with_emb[strand][:, :2], scores_only[strand], rtol=1e-5, atol=1e-6
        )
        assert np.isfinite(with_emb[strand]).all()
