import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from marin_dna.blog_review_sync import (
    DocumentReceipt,
    JsonlSyncLedger,
    ReviewBundle,
    SyncRequest,
    WebsiteReceipt,
    build_review_snapshot,
    coordinate_sync,
    prepare_review_bundle,
    request_from_bundle,
    snapshot_markdown,
)
from marin_dna.blog_workspace import default_config_path, load_config


SOURCE_SHA = "a" * 40
REQUESTED_AT = "2026-07-28T12:34:56Z"
REQUEST_ID = "issue-408-20260728T123456Z-a" * 1


def test_snapshot_preserves_article_structure_and_figure_order() -> None:
    config = load_config(default_config_path())
    snapshot = build_review_snapshot(
        config,
        source_sha=SOURCE_SHA,
        request_id=REQUEST_ID,
        requested_at=REQUESTED_AT,
    )

    assert snapshot.metadata.title == (
        "Building efficient and balanced genomic language models"
    )
    assert len(snapshot.figures) == 19
    assert [figure.number for figure in snapshot.figures] == list(range(1, 20))
    assert snapshot.figures[0].source_path.name == (
        "data_provenance_training_datasets.svg"
    )
    assert snapshot.figures[-1].source_path.name == (
        "figure11_leaderboard_heatmap__mendelian_probe.svg"
    )
    assert snapshot.body_markdown.count("[[MARIN_DNA_FIGURE_") == 19
    assert len(snapshot.footnotes) == 27
    assert [item.number for item in snapshot.footnotes] == list(range(1, 28))
    assert snapshot.body_markdown.count("MARINDNAFOOTNOTEREF") == 27
    assert "<figure" not in snapshot.body_markdown
    assert "<details" not in snapshot.body_markdown
    assert "## Notes" not in snapshot.body_markdown
    assert "[^glm-architecture]" not in snapshot.body_markdown
    assert "MARINDNAFOOTNOTEREF001" in snapshot.body_markdown
    assert "[MarinDNA repository](https://github.com/Open-Athena/marin-dna)" in (
        snapshot.body_markdown
    )
    companion = snapshot_markdown(snapshot)
    assert "[^glm-architecture]" in companion
    assert "[^glm-architecture]:" in companion
    assert "MARINDNAFOOTNOTEREF" not in companion


def _fake_figure_renderer(figure: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"\x89PNG\r\n\x1a\n" + repr(figure).encode())


def _fake_docx_renderer(
    snapshot: object, figures: tuple[object, ...], destination: Path
) -> None:
    destination.write_bytes(
        b"PK\x03\x04" + repr(snapshot).encode() + repr(figures).encode()
    )


def test_prepare_bundle_is_immutable_and_idempotent(tmp_path: Path) -> None:
    config = load_config(default_config_path())
    destination = tmp_path / REQUEST_ID
    first = prepare_review_bundle(
        config,
        source_sha=SOURCE_SHA,
        request_id=REQUEST_ID,
        requested_at=REQUESTED_AT,
        destination=destination,
        figure_renderer=_fake_figure_renderer,
        docx_renderer=_fake_docx_renderer,
    )
    original_manifest = first.manifest_path.read_bytes()

    second = prepare_review_bundle(
        config,
        source_sha=SOURCE_SHA,
        request_id=REQUEST_ID,
        requested_at=REQUESTED_AT,
        destination=destination,
        figure_renderer=lambda *_: pytest.fail("figure rerendered"),
        docx_renderer=lambda *_: pytest.fail("DOCX rerendered"),
    )

    assert second.root == first.root
    assert second.manifest_path.read_bytes() == original_manifest
    assert len(second.figures) == 19
    with pytest.raises(AssertionError, match="source SHA bundle conflict"):
        prepare_review_bundle(
            config,
            source_sha="b" * 40,
            request_id=REQUEST_ID,
            requested_at=REQUESTED_AT,
            destination=destination,
            figure_renderer=_fake_figure_renderer,
            docx_renderer=_fake_docx_renderer,
        )

    first.docx_path.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="review bundle file changed"):
        prepare_review_bundle(
            config,
            source_sha=SOURCE_SHA,
            request_id=REQUEST_ID,
            requested_at=REQUESTED_AT,
            destination=destination,
            figure_renderer=_fake_figure_renderer,
            docx_renderer=_fake_docx_renderer,
        )


@dataclass
class FakeWebsite:
    calls: int = 0
    failures_remaining: int = 0

    def publish(self, bundle: ReviewBundle, request: SyncRequest) -> WebsiteReceipt:
        assert bundle.manifest_path.is_file()
        assert request.request_id == REQUEST_ID
        self.calls += 1
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("website unavailable")
        return WebsiteReceipt(
            commit_sha="c" * 40,
            preview_url=(
                "https://cms-blog-genomic-lm-optimiza.openathena-ai.pages.dev/"
                "blog/marin-dna/"
            ),
            build_status="success",
        )


@dataclass
class FakeDocument:
    calls: int = 0
    failures_remaining: int = 0

    def publish(self, bundle: ReviewBundle, request: SyncRequest) -> DocumentReceipt:
        assert bundle.docx_path.is_file()
        assert request.request_id == REQUEST_ID
        self.calls += 1
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("Drive unavailable")
        return DocumentReceipt(
            document_url="https://docs.google.com/document/d/new-review/edit",
            revision_id="revision-1",
            verification_status="success",
        )


def _prepared_bundle(tmp_path: Path) -> tuple[ReviewBundle, SyncRequest]:
    config = load_config(default_config_path())
    bundle = prepare_review_bundle(
        config,
        source_sha=SOURCE_SHA,
        request_id=REQUEST_ID,
        requested_at=REQUESTED_AT,
        destination=tmp_path / REQUEST_ID,
        figure_renderer=_fake_figure_renderer,
        docx_renderer=_fake_docx_renderer,
    )
    return bundle, request_from_bundle(bundle)


def test_sync_retry_reuses_successful_artifacts(tmp_path: Path) -> None:
    bundle, request = _prepared_bundle(tmp_path)
    ledger = JsonlSyncLedger(tmp_path / "sync.jsonl")
    website = FakeWebsite()
    document = FakeDocument()

    first = coordinate_sync(
        bundle,
        request,
        ledger=ledger,
        website=website,
        document=document,
    )
    second = coordinate_sync(
        bundle,
        request,
        ledger=ledger,
        website=website,
        document=document,
    )

    assert first.complete and second.complete
    assert first.website == second.website
    assert first.document == second.document
    assert website.calls == 1
    assert document.calls == 1
    assert [event.target for event in ledger.read()] == [
        "request",
        "website",
        "document",
    ]


def test_sync_recovers_only_failed_target(tmp_path: Path) -> None:
    bundle, request = _prepared_bundle(tmp_path)
    ledger = JsonlSyncLedger(tmp_path / "sync.jsonl")
    website = FakeWebsite(failures_remaining=1)
    document = FakeDocument()

    partial = coordinate_sync(
        bundle,
        request,
        ledger=ledger,
        website=website,
        document=document,
    )
    recovered = coordinate_sync(
        bundle,
        request,
        ledger=ledger,
        website=website,
        document=document,
    )

    assert not partial.complete
    assert partial.website is None
    assert partial.website_error == "RuntimeError: website unavailable"
    assert partial.document is not None
    assert recovered.complete
    assert website.calls == 2
    assert document.calls == 1
    assert [event.status for event in ledger.read()] == [
        "registered",
        "failed",
        "succeeded",
        "succeeded",
    ]


def test_sync_rejects_conflicting_request_identity(tmp_path: Path) -> None:
    bundle, request = _prepared_bundle(tmp_path)
    ledger = JsonlSyncLedger(tmp_path / "sync.jsonl")
    website = FakeWebsite()
    document = FakeDocument()
    coordinate_sync(
        bundle,
        request,
        ledger=ledger,
        website=website,
        document=document,
    )

    raw = json.loads(bundle.manifest_path.read_text())
    raw["source_sha"] = "b" * 40
    bundle.manifest_path.write_text(json.dumps(raw))
    conflict = SyncRequest(
        request_id=request.request_id,
        source_sha="b" * 40,
        requested_at=request.requested_at,
        bundle_sha256=__import__("hashlib")
        .sha256(bundle.manifest_path.read_bytes())
        .hexdigest(),
    )
    with pytest.raises(AssertionError, match="request ID conflict"):
        coordinate_sync(
            bundle,
            conflict,
            ledger=ledger,
            website=website,
            document=document,
        )
