from dataclasses import replace
from pathlib import Path

import pytest

from marin_dna import blog_workspace
from marin_dna.blog_workspace import (
    default_config_path,
    export_workspace,
    extract_local_asset_references,
    inject_live_reload,
    load_config,
    materialize_article_preview,
    preview_source_signature,
    read_baseline_manifest,
    refresh_live_preview,
    validate_footnotes,
    validate_workspace,
)


def test_extract_local_asset_references() -> None:
    markdown = """
![local](/assets/images/blog/post/figure.svg)
![relative](other.svg "title")
![remote](https://example.com/remote.png)
<img src='/assets/images/blog/post/html.svg'>
{{plotly: chart.json | title="Chart"}}
"""
    assert extract_local_asset_references(markdown) == [
        "/assets/images/blog/post/figure.svg",
        "other.svg",
        "/assets/images/blog/post/html.svg",
        "chart.json",
    ]


def test_validate_footnotes_rejects_duplicate_definitions() -> None:
    markdown = "reference[^same]\n\n[^same]: first\n[^same]: second\n"
    with pytest.raises(AssertionError, match="duplicate footnote identifiers: same"):
        validate_footnotes(markdown)


def test_validate_footnotes_rejects_undefined_reference() -> None:
    with pytest.raises(AssertionError, match="undefined footnote identifiers: missing"):
        validate_footnotes("reference[^missing]\n")


def test_edited_workspace_is_valid_and_baseline_manifest_is_readable() -> None:
    config = load_config(default_config_path())
    referenced_assets = validate_workspace(config)
    manifest = read_baseline_manifest(config)

    assert len(referenced_assets) == 20
    assert len(manifest) == 12


def test_export_uses_exact_website_paths(tmp_path: Path) -> None:
    config = load_config(default_config_path())
    destination = tmp_path / "open-athena.github.io"
    exported = export_workspace(config, destination)

    expected = [destination / config.article_path]
    expected.extend(
        destination / path.relative_to(config.root)
        for path in sorted(config.assets.rglob("*"))
        if path.is_file()
    )
    assert exported == expected
    for target in exported:
        source = config.root / target.relative_to(destination)
        assert target.read_bytes() == source.read_bytes()


def test_preview_contains_only_article_and_referenced_files(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    page = dist / "blog/post/index.html"
    page.parent.mkdir(parents=True)
    page.write_text(
        '<link href="/css/style.css"><img src="/assets/article.svg">'
        '<a href="/blog/other/">Other article</a>'
    )
    style = dist / "css/style.css"
    style.parent.mkdir()
    style.write_text(
        '@font-face { src: url("../assets/font.woff2"); } '
        "body { background: url(/assets/background.svg); }"
    )
    assets = dist / "assets"
    assets.mkdir()
    for name in ("article.svg", "font.woff2", "background.svg", "unrelated.svg"):
        (assets / name).write_text(f"asset: {name}")
    other = dist / "blog/other/index.html"
    other.parent.mkdir(parents=True)
    other.write_text("private other article")

    output = tmp_path / "workspace/.preview"
    workspace_root = output.parent
    workspace_root.mkdir()
    copied = materialize_article_preview(
        dist, output, Path("blog/post/index.html"), workspace_root
    )

    assert copied == [
        Path("assets/article.svg"),
        Path("assets/background.svg"),
        Path("assets/font.woff2"),
        Path("blog/post/index.html"),
        Path("css/style.css"),
    ]
    assert list(output.rglob("*.html")) == [output / "blog/post/index.html"]
    assert not (output / "assets/unrelated.svg").exists()


def test_inject_live_reload_adds_preview_only_client(tmp_path: Path) -> None:
    output = tmp_path / ".preview"
    page_relative = Path("blog/post/index.html")
    page = output / page_relative
    page.parent.mkdir(parents=True)
    page.write_text("<html><body>draft</body></html>")

    inject_live_reload(output, page_relative, revision="revision-1")

    assert "__marin_dna_live_reload__" in page.read_text()
    assert (output / "__marin_dna_live_reload__").read_text() == "revision-1\n"


def test_preview_source_signature_tracks_article_and_assets(tmp_path: Path) -> None:
    config = replace(load_config(default_config_path()), root=tmp_path)
    config.article.parent.mkdir(parents=True)
    config.article.write_text("first draft")
    config.assets.mkdir(parents=True)
    asset = config.assets / "figure.svg"
    asset.write_text("<svg/>")

    initial = preview_source_signature(config)
    config.article.write_text("a longer second draft")
    after_article_edit = preview_source_signature(config)
    assert after_article_edit != initial

    (tmp_path / "unrelated.txt").write_text("ignored")
    assert preview_source_signature(config) == after_article_edit

    asset.write_text("<svg><title>updated</title></svg>")
    assert preview_source_signature(config) != after_article_edit


def test_refresh_live_preview_replaces_previous_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(load_config(default_config_path()), root=tmp_path)
    output = tmp_path / ".preview"
    output.mkdir()
    (output / "old.txt").write_text("old preview")

    def fake_build(
        _config: blog_workspace.WorkspaceConfig, destination: Path | None = None
    ) -> Path:
        assert destination is not None
        page = destination / "blog" / config.slug / "index.html"
        page.parent.mkdir(parents=True)
        page.write_text("<html><body>new preview</body></html>")
        return destination

    monkeypatch.setattr(blog_workspace, "build_preview", fake_build)
    refreshed = refresh_live_preview(config, revision="revision-2", output=output)

    assert refreshed == output
    assert not (output / "old.txt").exists()
    assert (output / "__marin_dna_live_reload__").read_text() == "revision-2\n"
    assert (
        "__marin_dna_live_reload__"
        in (output / "blog" / config.slug / "index.html").read_text()
    )


def test_refresh_live_preview_preserves_last_good_build_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(load_config(default_config_path()), root=tmp_path)
    output = tmp_path / ".preview"
    output.mkdir()
    marker = output / "last-good.txt"
    marker.write_text("last good preview")

    def failing_build(
        _config: blog_workspace.WorkspaceConfig, destination: Path | None = None
    ) -> Path:
        assert destination is not None
        destination.mkdir()
        (destination / "partial.txt").write_text("partial build")
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(blog_workspace, "build_preview", failing_build)
    with pytest.raises(RuntimeError, match="renderer failed"):
        refresh_live_preview(config, revision="revision-3", output=output)

    assert marker.read_text() == "last good preview"
    assert not output.with_name(".preview.next").exists()
