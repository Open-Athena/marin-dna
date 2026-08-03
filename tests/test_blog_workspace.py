import ast
from dataclasses import replace
from email.utils import formatdate
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import re
import threading
import time
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import pytest

from marin_dna import blog_workspace
from marin_dna.blog_figure_typography import (
    FIGURE_BASE_SIZE_PX,
    FIGURE_GLOBAL_RENDER_SCALE,
    MATPLOTLIB_BASE_SIZE,
    MATPLOTLIB_NOTE_SIZE,
    MATPLOTLIB_SMALL_SIZE_RATIO,
    MATPLOTLIB_TITLE_SIZE_RATIO,
    matplotlib_typography_rcparams,
    normalize_matplotlib_svg_typography,
    normalize_svg_typography,
    validate_svg_typography,
)
from marin_dna.blog_workspace import (
    default_config_path,
    export_workspace,
    extract_local_asset_references,
    extract_svg_render_widths,
    inject_live_reload,
    load_config,
    materialize_article_preview,
    preview_source_signature,
    read_baseline_manifest,
    refresh_live_preview,
    validate_footnotes,
    validate_workspace,
)


ACTIVE_BLOG_PLOT_RECIPES = (
    "plots/blog/promoter_cds_specialists.py",
    "plots/upstream_cds_balance.py",
    "plots/blog/_leaderboard.py",
    "plots/blog/marin_dna/src/figures/figure1_lr_transfer.py",
    "plots/blog/marin_dna/src/figures/figure2_beta2_epsilon_transfer.py",
    "plots/blog/marin_dna/src/figures/figure3_region_hyper_transfer.py",
    "plots/blog/marin_dna/src/figures/figure4_loss_scaling.py",
    "plots/blog/marin_dna/src/figures/figure5_params_vs_vep_auprc.py",
    "plots/blog/marin_dna/src/figures/figure6_loss_vs_vep_auprc.py",
    "plots/blog/marin_dna/src/figures/figure6b_marin_evo2_missense.py",
    "plots/blog/marin_dna/src/figures/figure16_offline_lineage_prototype.py",
    "plots/blog/marin_dna/src/utils/figure_style.py",
    "plots/blog/marin_dna/src/utils/sweep_panel.py",
)

ACTIVE_DATA_FIGURE_ASSETS = (
    "promoter_cds_specialists.svg",
    "upstream_cds_balance.svg",
    "figure1_lr_transfer.svg",
    "figure2_beta2_epsilon_transfer.svg",
    "figure3_region_hyper_transfer.svg",
    "figure4_loss_scaling.svg",
    "figure5_params_vs_vep_auprc.svg",
    "figure6_loss_vs_vep_auprc.svg",
    "figure6b_marin_evo2_missense.svg",
    "figure16_offline_lineage_llr_prototype.svg",
    "figure16_offline_lineage_probe_prototype.svg",
    "figure11_leaderboard_heatmap__mendelian_llr.svg",
    "figure11_leaderboard_heatmap__mendelian_probe.svg",
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


def test_validate_svg_intrinsic_dimensions_rejects_viewbox_only_svg(
    tmp_path: Path,
) -> None:
    svg = tmp_path / "figure.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 440"/>')

    with pytest.raises(AssertionError, match="lacks intrinsic width"):
        blog_workspace.validate_svg_intrinsic_dimensions(svg)


def test_validate_svg_intrinsic_dimensions_rejects_mismatched_aspect_ratio(
    tmp_path: Path,
) -> None:
    svg = tmp_path / "figure.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="400" '
        'viewBox="0 0 960 440"/>'
    )

    with pytest.raises(AssertionError, match="different aspect ratios"):
        blog_workspace.validate_svg_intrinsic_dimensions(svg)


def test_normalize_svg_typography_is_idempotent_and_preserves_monospace() -> None:
    original = """
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="440"
     viewBox="0 0 960 440" font-family="system-ui, sans-serif">
  <text font-size="9">tick</text>
  <text style="font-size: 20px; font-family: 'DejaVu Sans'">Title</text>
  <g font-size="10" font-family="ui-monospace, monospace"><text>ACGT</text></g>
</svg>
"""
    normalized = normalize_svg_typography(original)

    assert normalize_svg_typography(normalized) == normalized
    assert 'data-figure-typography="lato-v1"' in normalized
    assert 'data-figure-render-width="700"' in normalized
    assert 'font-family="Lato, sans-serif"' in normalized
    assert "font-family: 'Lato', sans-serif" in normalized
    assert 'font-family="ui-monospace, monospace"' in normalized


def test_matplotlib_typography_uses_one_base_and_explicit_ratios() -> None:
    params = matplotlib_typography_rcparams()

    assert FIGURE_GLOBAL_RENDER_SCALE == 1.2
    assert MATPLOTLIB_TITLE_SIZE_RATIO == 1.2
    assert MATPLOTLIB_SMALL_SIZE_RATIO == 5.0 / 6.0
    assert FIGURE_BASE_SIZE_PX == MATPLOTLIB_BASE_SIZE * FIGURE_GLOBAL_RENDER_SCALE
    assert MATPLOTLIB_NOTE_SIZE == MATPLOTLIB_BASE_SIZE * (5.0 / 6.0)
    assert params["axes.titlesize"] == MATPLOTLIB_BASE_SIZE * 1.2
    assert params["figure.titlesize"] == MATPLOTLIB_BASE_SIZE * 1.2
    for role in (
        "font.size",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "legend.title_fontsize",
    ):
        assert params[role] == MATPLOTLIB_BASE_SIZE


def test_matplotlib_svg_uses_the_one_shared_whole_figure_scale() -> None:
    original = """
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="440"
     viewBox="0 0 960 440" font-family="DejaVu Sans">
  <text font-size="10">Body</text>
  <text font-size="12">Title</text>
</svg>
"""
    normalized = normalize_matplotlib_svg_typography(original)

    assert 'data-figure-render-scale="1.2"' in normalized
    assert 'data-figure-render-width="1152"' in normalized
    assert 'font-size="10"' in normalized
    assert 'font-size="12"' in normalized
    assert normalize_svg_typography(normalized, render_width_px=1152) == normalized


def test_all_active_data_figures_use_the_one_shared_scale() -> None:
    root = Path(__file__).resolve().parents[1]
    asset_dir = (
        root
        / "blog"
        / "marin-dna"
        / "static"
        / "assets"
        / "images"
        / "blog"
        / "marin-dna"
    )
    for name in ACTIVE_DATA_FIGURE_ASSETS:
        svg_root = ElementTree.parse(asset_dir / name).getroot()
        assert float(svg_root.attrib["data-figure-render-scale"]) == (
            FIGURE_GLOBAL_RENDER_SCALE
        ), name


def test_all_blog_figure_captions_lead_with_a_bold_number_and_title() -> None:
    root = Path(__file__).resolve().parents[1]
    article = (
        root
        / "blog"
        / "marin-dna"
        / "content"
        / "blog"
        / "marin-dna.md"
    ).read_text(encoding="utf-8")
    captions = re.findall(r"<figcaption>(.*?)</figcaption>", article, re.DOTALL)

    assert len(captions) == 20
    for number, caption in enumerate(captions, start=1):
        assert re.fullmatch(
            rf"<strong>Figure {number}: [^<]+\.</strong>(?:\s+.+)?",
            caption,
            re.DOTALL,
        ), caption


def test_blog_figure_references_use_compact_linked_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    article = (
        root
        / "blog"
        / "marin-dna"
        / "content"
        / "blog"
        / "marin-dna.md"
    ).read_text(encoding="utf-8")
    figure_numbers = {
        figure_id: int(number)
        for figure_id, number in re.findall(
            r'<figure id="([^"]+)"[^>]*>.*?'
            r"<figcaption><strong>Figure (\d+):",
            article,
            flags=re.DOTALL,
        )
    }
    markdown_references = [
        (figure_id, label)
        for label, figure_id in re.findall(r"\[([^]]+)\]\(#(fig-[^)]+)\)", article)
    ]
    html_references = re.findall(r'<a href="#(fig-[^"]+)">([^<]+)</a>', article)

    assert figure_numbers
    for figure_id, label in markdown_references + html_references:
        assert figure_id in figure_numbers, figure_id
        assert label == f"Fig. {figure_numbers[figure_id]}", (figure_id, label)

    outside_captions = re.sub(
        r"<figcaption>.*?</figcaption>", "", article, flags=re.DOTALL
    )
    assert not re.search(r"\bFigure \d+\b", outside_captions)


def test_active_data_plot_recipes_omit_overall_figure_titles() -> None:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for relative_path in ACTIVE_BLOG_PLOT_RECIPES:
        path = root / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "suptitle"
            ):
                failures.append(f"{relative_path}:{node.lineno}")
    assert not failures, "overall titles belong in captions:\n" + "\n".join(failures)


def test_active_plot_recipes_cannot_override_standard_element_font_sizes() -> None:
    root = Path(__file__).resolve().parents[1]
    standard_calls = {
        "legend": {"fontsize", "title_fontsize"},
        "set_title": {"fontsize"},
        "suptitle": {"fontsize"},
        "set_xlabel": {"fontsize"},
        "set_ylabel": {"fontsize"},
        "set_xticklabels": {"fontsize"},
        "set_yticklabels": {"fontsize"},
        "tick_params": {"labelsize"},
    }
    failures: list[str] = []
    for relative_path in ACTIVE_BLOG_PLOT_RECIPES:
        path = root / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                call_name = node.func.id
            else:
                continue
            forbidden = standard_calls.get(call_name, set())
            present = sorted(
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None and keyword.arg in forbidden
            )
            if present:
                failures.append(f"{relative_path}:{node.lineno}: {present}")
    assert not failures, (
        "standard typography must come from shared rcParams:\n" + "\n".join(failures)
    )


def test_active_plot_recipes_cannot_override_primary_glyph_sizes() -> None:
    root = Path(__file__).resolve().parents[1]
    standard_calls = {
        "plot": {"linewidth", "markersize", "markeredgewidth"},
        "errorbar": {
            "linewidth",
            "elinewidth",
            "markersize",
            "markeredgewidth",
        },
        "scatter": {"s", "linewidths"},
        "Line2D": {"linewidth", "markersize", "markeredgewidth"},
    }
    failures: list[str] = []
    for relative_path in ACTIVE_BLOG_PLOT_RECIPES:
        path = root / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                call_name = node.func.id
            else:
                continue
            forbidden = standard_calls.get(call_name, set())
            for keyword in node.keywords:
                if keyword.arg not in forbidden:
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(
                    value.value, int | float
                ):
                    failures.append(
                        f"{relative_path}:{node.lineno}: {keyword.arg}={value.value}"
                    )
    assert not failures, (
        "primary glyph sizes must inherit Matplotlib defaults:\n" + "\n".join(failures)
    )


def test_validate_svg_typography_rejects_unscaled_text(tmp_path: Path) -> None:
    svg = tmp_path / "figure.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="440" '
        'viewBox="0 0 960 440" data-figure-typography="lato-v1" '
        'data-figure-render-width="700" '
        'font-family="Lato, sans-serif"><text font-size="9">tiny</text></svg>'
    )

    with pytest.raises(AssertionError, match="outside the 10–16px hierarchy"):
        validate_svg_typography(svg)


def test_normalize_svg_typography_preserves_marked_emoji(tmp_path: Path) -> None:
    original = """
<svg xmlns="http://www.w3.org/2000/svg" width="840" height="350"
     viewBox="0 0 840 350" font-family="system-ui, sans-serif">
  <text font-size="16">Label</text>
  <text data-figure-preserve-typography="true" font-size="48"
        font-family="Noto Color Emoji, Apple Color Emoji, sans-serif">❄️</text>
</svg>
"""
    normalized = normalize_svg_typography(original, render_width_px=640)

    assert normalize_svg_typography(normalized, render_width_px=640) == normalized
    assert 'data-figure-preserve-typography="true" font-size="48"' in normalized
    assert 'font-family="Noto Color Emoji, Apple Color Emoji, sans-serif"' in normalized

    svg = tmp_path / "figure.svg"
    svg.write_text(normalized)
    validate_svg_typography(svg, expected_render_width_px=640)


def test_extract_svg_render_widths_accounts_for_frame_padding() -> None:
    markdown = """
<figure id="compact" data-figure-width="600">
<img src="/assets/images/blog/post/compact.svg?v=2" />
<figcaption>Compact figure</figcaption>
</figure>
<figure id="raster" data-figure-width="500">
<img src="/assets/images/blog/post/photo.png" />
</figure>
"""

    assert extract_svg_render_widths(markdown) == {
        "/assets/images/blog/post/compact.svg": 560.0
    }


def test_extract_svg_render_widths_accepts_compact_plot() -> None:
    markdown = """
<figure id="compact" data-figure-width="276.92">
<img src="/assets/images/blog/post/compact.svg" />
</figure>
"""

    widths = extract_svg_render_widths(markdown)
    assert widths == {"/assets/images/blog/post/compact.svg": pytest.approx(236.92)}


def test_extract_svg_render_widths_requires_declared_width() -> None:
    markdown = '<figure><img src="/assets/images/blog/post/figure.svg" /></figure>'

    with pytest.raises(AssertionError, match="lacks data-figure-width"):
        extract_svg_render_widths(markdown)


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


def test_preview_request_handler_disables_conditional_cache(tmp_path: Path) -> None:
    page = tmp_path / "index.html"
    page.write_text("fresh preview")
    handler = partial(blog_workspace.PreviewRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        future = formatdate(time.time() + 3600, usegmt=True)
        request = Request(
            f"http://127.0.0.1:{server.server_port}/index.html",
            headers={"If-Modified-Since": future},
        )
        with urlopen(request) as response:
            assert response.status == 200
            assert response.read() == b"fresh preview"
            assert response.headers["Cache-Control"] == (
                "no-store, no-cache, must-revalidate, max-age=0"
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


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
