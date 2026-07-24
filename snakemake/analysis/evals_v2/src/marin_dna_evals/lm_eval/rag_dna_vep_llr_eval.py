"""Paired cached-prefix lm-eval task for issue #402's frozen RAG rows."""

from __future__ import annotations

from typing import Any

from lm_eval.api.instance import Instance

from marin_dna_evals.lm_eval.dna_vep_llr_eval import DnaVepLlrEvalTask
from marin_dna.pipelines.rag_glm.lm_eval_adapter import (
    install_levanter_rag_loglikelihood,
)

install_levanter_rag_loglikelihood()


class RagDnaVepLlrEvalTask(DnaVepLlrEvalTask):
    """Keep both completions in one request so the adapter must share the prefix."""

    VERSION = 1

    def construct_requests(self, doc: Any, ctx: str, **kwargs: Any) -> Instance:
        metadata = kwargs.get("metadata", (None, None, None))
        return Instance(
            request_type="rag_loglikelihood",
            doc=doc,
            arguments=(ctx, doc["ref_completion"], doc["alt_completion"]),
            idx=0,
            metadata=metadata,
        )

    def process_results(
        self, doc: Any, results: list[tuple[float, float, float]]
    ) -> dict[str, tuple[Any, ...]]:
        assert len(results) == 1
        ref_loglikelihood, alt_loglikelihood, raw_llr = results[0]
        assert abs(raw_llr - (alt_loglikelihood - ref_loglikelihood)) < 1.0e-4
        variant_id = (
            str(doc["chrom"]),
            int(doc["pos"]),
            str(doc["ref"]),
            str(doc["alt"]),
        )
        strand = str(doc.get("strand", "+"))
        return {
            metric: (
                float(raw_llr),
                doc["target"],
                doc.get("subset"),
                variant_id,
                int(doc["match_group"]),
                strand,
            )
            for metric in self._metrics
        }
