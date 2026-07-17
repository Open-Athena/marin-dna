from pathlib import Path

import pytest

from marin_dna.blog_workspace import (
    default_config_path,
    export_workspace,
    extract_local_asset_references,
    load_config,
    read_baseline_manifest,
    sha256_file,
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


def test_imported_workspace_is_valid_and_matches_manifest() -> None:
    config = load_config(default_config_path())
    referenced_assets = validate_workspace(config)
    manifest = read_baseline_manifest(config)

    assert len(referenced_assets) == 11
    assert len(manifest) == 12
    for relative, expected_digest in manifest.items():
        assert sha256_file(config.root / relative) == expected_digest


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
