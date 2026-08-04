from dataclasses import dataclass, field
from pathlib import Path

from marin_dna.blog_review_sync import (
    DocumentReceipt,
    JsonlSyncLedger,
    ReviewBundle,
    ReviewFigure,
    SyncRequest,
    WebsiteReceipt,
    coordinate_sync,
    prepare_review_bundle,
    request_from_bundle,
)
from marin_dna.blog_workspace import default_config_path, load_config


SOURCE_SHA = "d" * 40
REQUESTED_AT = "2026-07-28T20:00:00Z"
PREVIEW_URL = (
    "https://cms-blog-genomic-lm-optimiza.openathena-ai.pages.dev/blog/marin-dna/"
)


def _render_figure(_figure: ReviewFigure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"\x89PNG\r\n\x1a\nfixture")


def _render_docx(
    _snapshot: object, _figures: tuple[object, ...], destination: Path
) -> None:
    destination.write_bytes(b"PK\x03\x04fixture")


@dataclass
class CountingWebsite:
    request_ids: list[str] = field(default_factory=list)

    def publish(self, _bundle: ReviewBundle, request: SyncRequest) -> WebsiteReceipt:
        self.request_ids.append(request.request_id)
        return WebsiteReceipt(
            commit_sha="e" * 40,
            preview_url=PREVIEW_URL,
            build_status="success",
        )


@dataclass
class CountingDocuments:
    request_ids: list[str] = field(default_factory=list)

    def publish(self, _bundle: ReviewBundle, request: SyncRequest) -> DocumentReceipt:
        self.request_ids.append(request.request_id)
        return DocumentReceipt(
            document_url=(
                f"https://docs.google.com/document/d/{request.request_id}/edit"
            ),
            revision_id=f"revision-{request.request_id}",
            verification_status="success",
        )


def test_distinct_requests_for_same_source_create_distinct_docs(
    tmp_path: Path,
) -> None:
    config = load_config(default_config_path())
    ledger = JsonlSyncLedger(tmp_path / "sync.jsonl")
    website = CountingWebsite()
    documents = CountingDocuments()
    reports = []

    for request_id in ("issue-408-request-one", "issue-408-request-two"):
        bundle = prepare_review_bundle(
            config,
            source_sha=SOURCE_SHA,
            request_id=request_id,
            requested_at=REQUESTED_AT,
            destination=tmp_path / request_id,
            figure_renderer=_render_figure,
            docx_renderer=_render_docx,
        )
        reports.append(
            coordinate_sync(
                bundle,
                request_from_bundle(bundle),
                ledger=ledger,
                website=website,
                document=documents,
            )
        )

    assert all(report.complete for report in reports)
    assert website.request_ids == ["issue-408-request-one", "issue-408-request-two"]
    assert documents.request_ids == ["issue-408-request-one", "issue-408-request-two"]
    assert reports[0].document != reports[1].document
