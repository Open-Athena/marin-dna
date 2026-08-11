"""Paired cached-prefix lm-eval task for issue #402's frozen RAG rows."""

from __future__ import annotations

import logging
from typing import Any

import datasets
from lm_eval.api.instance import Instance

from marin_dna_evals.lm_eval.dna_vep_llr_eval import DnaVepLlrEvalTask
from marin_dna_evals.rag_glm.lm_eval_adapter import (
    install_levanter_rag_loglikelihood,
)

install_levanter_rag_loglikelihood()

_logger = logging.getLogger(__name__)
_DEFAULT_CACHE_DIR = "/tmp/marin-dna-rag-evals"


class RagDnaVepLlrEvalTask(DnaVepLlrEvalTask):
    """Keep both completions in one request so the adapter must share the prefix."""

    VERSION = 1

    def __init__(
        self,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        download_mode: Any = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        try:
            super().__init__(
                data_dir=data_dir,
                cache_dir=cache_dir,
                download_mode=download_mode,
                config=config,
            )
        except Exception:
            _logger.exception(
                "RAG task initialization failed (dataset=%r revision=%r cache_dir=%r)",
                self.DATASET_PATH,
                self.DATASET_REVISION,
                cache_dir or _DEFAULT_CACHE_DIR,
            )
            raise

    def download(
        self,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        download_mode: Any = None,
    ) -> None:
        """Load the frozen Parquets directly instead of rediscovering the repo.

        Fresh Iris workers occasionally fail in Hugging Face dataset discovery
        before any training step. These issue-402 repositories have an asserted
        two-file layout, so commit-pinned resolve URLs are both simpler and more
        deterministic.
        """
        assert self.DATASET_PATH, "RAG eval task requires dataset_path"
        assert self.DATASET_REVISION, "RAG eval task requires dataset_revision"
        base_url = (
            f"https://huggingface.co/datasets/{self.DATASET_PATH}/resolve/"
            f"{self.DATASET_REVISION}"
        )
        data_files = {
            split: f"{base_url}/{split}.parquet" for split in ("train", "test")
        }
        load_kwargs: dict[str, Any] = {
            "path": "parquet",
            "data_files": data_files,
            "cache_dir": cache_dir or _DEFAULT_CACHE_DIR,
        }
        if data_dir is not None:
            load_kwargs["data_dir"] = data_dir
        if download_mode is not None:
            load_kwargs["download_mode"] = download_mode
        self.dataset = datasets.load_dataset(
            **load_kwargs,
        )

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
