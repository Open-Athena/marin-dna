from pathlib import Path

import pytest

from marin_dna.blog_workspace import (
    default_config_path,
    export_workspace,
    extract_local_asset_references,
    load_config,
    materialize_article_preview,
    read_baseline_manifest,
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

    assert len(referenced_assets) == 19
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
