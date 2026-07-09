"""LoRA supervised fine-tuning of a frozen gLM for variant effect prediction (#369).

Siamese LoRA-adapted backbone → entire-window mean-pool (fp32, BOS excluded) of the
ref and alt 255bp+BOS windows → ``concat_ref_delta = [pool_ref, pool_alt−pool_ref]``
→ linear head → BCE. The direct "unfreeze the frozen linear probe" comparison from
#341/#314: same labels, same pooled feature, but the backbone is now trainable.

Overfitting the ~5k labelled missense SNVs is the central risk, so the primary
instrument is the per-step train-vs-validation-chromosome AUPRC gap, and the primary
sweep axis is regularization (training length / capacity / weight decay / dropout).

Reuses the evals apparatus verbatim: ``Genome`` reads, ``transform_llr_clm`` windows
(``data.transforms``), the ``AutoModelForCausalLM`` load pattern, and
``per_chrom_weighted_ap`` scoring (``pipelines.evals.metrics``).
"""
