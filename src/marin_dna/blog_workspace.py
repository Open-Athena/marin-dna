"""Validation, preview, and export tooling for the issue #373 blog workspace."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


DEFAULT_WORKSPACE_RELATIVE_PATH = Path("blog/genomic-lm-optimization")
MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)"
)
HTML_IMAGE_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
PLOTLY_RE = re.compile(r"\{\{\s*plotly:\s*([^|}\s]+)", re.IGNORECASE)
FOOTNOTE_DEFINITION_RE = re.compile(r"(?m)^[ \t]{0,3}\[\^([^\]\s]+)\]:")
FOOTNOTE_TOKEN_RE = re.compile(r"\[\^([^\]\s]+)\]")


@dataclass(frozen=True)
class WorkspaceConfig:
    """Pinned source, renderer, and path configuration for one blog workspace."""

    root: Path
    slug: str
    website_repository: str
    website_pr: int
    website_pr_commit: str
    renderer_commit: str
    blog_source_repository: str
    blog_source_commit: str
    article_path: Path
    assets_path: Path
    baseline_manifest: Path
    renderer_lock: Path
    baseline_page_sha256: str

    @property
    def article(self) -> Path:
        return self.root / self.article_path

    @property
    def assets(self) -> Path:
        return self.root / self.assets_path

    @property
    def static(self) -> Path:
        return self.root / "static"


def repository_root() -> Path:
    """Return the MarinDNA checkout root containing this module."""
    root = Path(__file__).resolve().parents[2]
    assert (root / "pyproject.toml").is_file(), root
    return root


def default_config_path() -> Path:
    """Return the issue #373 workspace configuration path."""
    return repository_root() / DEFAULT_WORKSPACE_RELATIVE_PATH / "workspace.toml"


def load_config(path: Path) -> WorkspaceConfig:
    """Load and defensively validate a workspace configuration file."""
    resolved = path.resolve()
    with resolved.open("rb") as stream:
        raw = tomllib.load(stream)

    required = {
        "slug",
        "website_repository",
        "website_pr",
        "website_pr_commit",
        "renderer_commit",
        "blog_source_repository",
        "blog_source_commit",
        "article_path",
        "assets_path",
        "baseline_manifest",
        "renderer_lock",
        "baseline_page_sha256",
    }
    assert set(raw) == required, f"workspace.toml keys differ: {set(raw) ^ required}"
    for key in ("website_pr_commit", "renderer_commit", "blog_source_commit"):
        assert re.fullmatch(r"[0-9a-f]{40}", raw[key]), f"invalid {key}"
    assert re.fullmatch(r"[0-9a-f]{64}", raw["baseline_page_sha256"]), (
        "invalid baseline_page_sha256"
    )

    root = resolved.parent
    article_path = _safe_relative_path(raw["article_path"])
    assets_path = _safe_relative_path(raw["assets_path"])
    baseline_manifest = _safe_relative_path(raw["baseline_manifest"])
    renderer_lock = _safe_relative_path(raw["renderer_lock"])
    assert article_path == Path("content/blog") / f"{raw['slug']}.md"
    assert assets_path == Path("static/assets/images/blog") / raw["slug"]
    return WorkspaceConfig(
        root=root,
        slug=raw["slug"],
        website_repository=raw["website_repository"],
        website_pr=raw["website_pr"],
        website_pr_commit=raw["website_pr_commit"],
        renderer_commit=raw["renderer_commit"],
        blog_source_repository=raw["blog_source_repository"],
        blog_source_commit=raw["blog_source_commit"],
        article_path=article_path,
        assets_path=assets_path,
        baseline_manifest=baseline_manifest,
        renderer_lock=renderer_lock,
        baseline_page_sha256=raw["baseline_page_sha256"],
    )


def _safe_relative_path(value: str) -> Path:
    """Parse a repository-relative path and reject traversal or absolute paths."""
    path = Path(value)
    assert not path.is_absolute(), value
    assert ".." not in path.parts, value
    assert path.parts, value
    return path


def extract_local_asset_references(markdown: str) -> list[str]:
    """Extract local image and Plotly asset URLs from article Markdown."""
    references: list[str] = []
    for match in MARKDOWN_IMAGE_RE.finditer(markdown):
        references.append(match.group(1) or match.group(2))
    references.extend(match.group(1) for match in HTML_IMAGE_RE.finditer(markdown))
    references.extend(match.group(1) for match in PLOTLY_RE.finditer(markdown))

    local: list[str] = []
    for reference in references:
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc or reference.startswith(("#", "data:")):
            continue
        path = unquote(parsed.path)
        assert path, f"empty local asset reference: {reference!r}"
        local.append(path)
    return local


def validate_footnotes(markdown: str) -> None:
    """Reject duplicate definitions and references without definitions."""
    definitions = FOOTNOTE_DEFINITION_RE.findall(markdown)
    duplicates = sorted(
        name for name, count in Counter(definitions).items() if count > 1
    )
    assert not duplicates, f"duplicate footnote identifiers: {', '.join(duplicates)}"

    definition_spans = {
        match.span() for match in FOOTNOTE_DEFINITION_RE.finditer(markdown)
    }
    references = {
        match.group(1)
        for match in FOOTNOTE_TOKEN_RE.finditer(markdown)
        if not any(
            start <= match.start() and match.end() <= end
            for start, end in definition_spans
        )
    }
    undefined = sorted(references - set(definitions))
    assert not undefined, f"undefined footnote identifiers: {', '.join(undefined)}"


def _asset_reference_to_static_path(reference: str, slug: str) -> Path:
    """Map one site-root or post-relative asset URL into the static tree."""
    if reference.startswith("/"):
        relative = PurePosixPath(reference.removeprefix("/"))
    elif reference.startswith("assets/"):
        relative = PurePosixPath(reference)
    else:
        relative = PurePosixPath("assets/images/blog") / slug / reference
    assert ".." not in relative.parts, f"asset path traversal: {reference}"
    return Path(*relative.parts)


def validate_workspace(config: WorkspaceConfig) -> list[Path]:
    """Validate the canonical article, footnotes, and referenced local assets."""
    assert config.article.is_file(), f"missing canonical article: {config.article}"
    assert config.assets.is_dir(), f"missing canonical asset directory: {config.assets}"
    markdown = config.article.read_text()
    validate_footnotes(markdown)

    referenced_paths: list[Path] = []
    for reference in extract_local_asset_references(markdown):
        relative = _asset_reference_to_static_path(reference, config.slug)
        source = config.static / relative
        assert source.is_file(), f"missing local asset {reference}: expected {source}"
        referenced_paths.append(source)
    assert referenced_paths, "article has no local assets"
    return referenced_paths


def read_baseline_manifest(config: WorkspaceConfig) -> dict[Path, str]:
    """Read the sorted SHA-256 manifest for the untouched import baseline."""
    manifest_path = config.root / config.baseline_manifest
    assert manifest_path.is_file(), f"missing baseline manifest: {manifest_path}"
    entries: dict[Path, str] = {}
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        digest, separator, path_text = line.partition("  ")
        assert separator and re.fullmatch(r"[0-9a-f]{64}", digest), (
            f"invalid manifest line {line_number}"
        )
        relative = _safe_relative_path(path_text)
        assert relative not in entries, f"duplicate manifest path: {relative}"
        entries[relative] = digest
    assert entries, "empty baseline manifest"
    assert list(entries) == sorted(entries), "baseline manifest must be path-sorted"
    return entries


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_baseline(config: WorkspaceConfig, renderer: Path) -> None:
    """Prove the baseline import matches its manifest and pinned website commit."""
    manifest = read_baseline_manifest(config)
    workspace_files = {config.article_path}
    workspace_files.update(
        path.relative_to(config.root)
        for path in config.assets.rglob("*")
        if path.is_file()
    )
    assert workspace_files == set(manifest), (
        f"baseline file set differs: {workspace_files ^ set(manifest)}"
    )

    for relative, expected_digest in manifest.items():
        workspace_path = config.root / relative
        renderer_path = renderer / relative
        assert workspace_path.is_file(), f"missing baseline file: {workspace_path}"
        assert renderer_path.is_file(), f"missing pinned website file: {renderer_path}"
        actual_digest = sha256_file(workspace_path)
        assert actual_digest == expected_digest, f"baseline digest changed: {relative}"
        assert workspace_path.read_bytes() == renderer_path.read_bytes(), (
            f"baseline differs from website PR commit: {relative}"
        )


def _run(command: list[str], *, cwd: Path) -> None:
    """Run one subprocess and propagate failures without shell interpretation."""
    subprocess.run(command, cwd=cwd, check=True)


def clone_renderer(config: WorkspaceConfig, destination: Path) -> None:
    """Clone and check out the exact website renderer commit."""
    assert not destination.exists(), destination
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            config.website_repository,
            str(destination),
        ],
        cwd=destination.parent,
    )
    _run(["git", "checkout", "--detach", config.renderer_commit], cwd=destination)
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == config.renderer_commit


def overlay_workspace(config: WorkspaceConfig, renderer: Path) -> None:
    """Overlay the single canonical article and its assets onto a renderer clone."""
    article_target = renderer / config.article_path
    assets_target = renderer / config.assets_path
    assert article_target.parent.is_dir(), article_target.parent
    assert assets_target.parent.is_dir(), assets_target.parent
    shutil.copy2(config.article, article_target)
    renderer_lock = config.root / config.renderer_lock
    assert renderer_lock.is_file(), renderer_lock
    shutil.copy2(renderer_lock, renderer / "uv.lock")
    if assets_target.exists():
        assert assets_target.name == config.slug
        shutil.rmtree(assets_target)
    shutil.copytree(config.assets, assets_target)


def _run_renderer_build(config: WorkspaceConfig, renderer: Path) -> Path:
    """Run the pinned website build and verify the expected post and assets exist."""
    _run(["uv", "run", "--frozen", "build.py"], cwd=renderer)
    dist = renderer / "dist"
    page = dist / "blog" / config.slug / "index.html"
    assert page.is_file(), f"renderer did not create {page}"
    assert (dist / "css" / "style.css").is_file(), "renderer CSS missing from build"
    assert (dist / "assets" / "fonts" / "Herbik-Regular.ttf").is_file(), (
        "renderer font missing from build"
    )
    for reference in extract_local_asset_references(config.article.read_text()):
        relative = _asset_reference_to_static_path(reference, config.slug)
        assert (dist / relative).is_file(), f"built asset missing: {relative}"
    return dist


def _replace_generated_tree(
    source: Path, destination: Path, workspace_root: Path
) -> None:
    """Replace one explicitly generated tree below the workspace root."""
    resolved_root = workspace_root.resolve()
    resolved_destination = destination.resolve()
    assert resolved_destination.is_relative_to(resolved_root)
    assert resolved_destination != resolved_root
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def build_preview(config: WorkspaceConfig, output: Path | None = None) -> Path:
    """Build the article with the pinned real renderer into a local output tree."""
    validate_workspace(config)
    output = output or config.root / ".preview"
    with tempfile.TemporaryDirectory(prefix="marin-dna-blog-renderer-") as temporary:
        renderer = Path(temporary) / "open-athena.github.io"
        clone_renderer(config, renderer)
        overlay_workspace(config, renderer)
        dist = _run_renderer_build(config, renderer)
        _replace_generated_tree(dist, output, config.root)
    return output


def verify_import(config: WorkspaceConfig) -> str:
    """Verify byte identity with the pinned PR and build the untouched baseline."""
    validate_workspace(config)
    assert config.renderer_commit == config.website_pr_commit
    with tempfile.TemporaryDirectory(prefix="marin-dna-blog-baseline-") as temporary:
        renderer = Path(temporary) / "open-athena.github.io"
        clone_renderer(config, renderer)
        verify_baseline(config, renderer)
        overlay_workspace(config, renderer)
        dist = _run_renderer_build(config, renderer)
        page = dist / "blog" / config.slug / "index.html"
        page_digest = sha256_file(page)
        assert page_digest == config.baseline_page_sha256, "baseline render changed"
        return page_digest


def export_workspace(config: WorkspaceConfig, destination: Path) -> list[Path]:
    """Export the canonical article/assets to their exact website repository paths."""
    validate_workspace(config)
    destination = destination.resolve()
    assert destination != Path(destination.anchor), (
        "refusing to export to filesystem root"
    )
    destination.mkdir(parents=True, exist_ok=True)

    article_target = destination / config.article_path
    assets_target = destination / config.assets_path
    assert article_target.is_relative_to(destination)
    assert assets_target.is_relative_to(destination)
    article_target.parent.mkdir(parents=True, exist_ok=True)
    assets_target.parent.mkdir(parents=True, exist_ok=True)

    temporary_article = article_target.with_name(f".{article_target.name}.export.tmp")
    shutil.copy2(config.article, temporary_article)
    os.replace(temporary_article, article_target)
    if assets_target.exists():
        assert assets_target.name == config.slug
        shutil.rmtree(assets_target)
    shutil.copytree(config.assets, assets_target)

    exported = [article_target]
    exported.extend(sorted(path for path in assets_target.rglob("*") if path.is_file()))
    source_asset_count = sum(1 for path in config.assets.rglob("*") if path.is_file())
    assert len(exported) == 1 + source_asset_count
    return exported


def serve_preview(config: WorkspaceConfig, host: str, port: int) -> None:
    """Build and serve the preview until interrupted."""
    output = build_preview(config)
    handler = partial(SimpleHTTPRequestHandler, directory=str(output))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Preview: http://{host}:{port}/blog/{config.slug}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("verify-import")
    subparsers.add_parser("build")
    preview = subparsers.add_parser("preview")
    preview.add_argument("--host", default="127.0.0.1")
    preview.add_argument("--port", type=int, default=8765)
    export = subparsers.add_parser("export")
    export.add_argument("--destination", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected workspace operation."""
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate":
        assets = validate_workspace(config)
        print(f"Validated article and {len(assets)} referenced assets")
    elif args.command == "verify-import":
        page_digest = verify_import(config)
        print(f"Baseline import and render verified; page sha256={page_digest}")
    elif args.command == "build":
        output = build_preview(config)
        print(f"Built preview at {output}")
    elif args.command == "preview":
        assert 0 < args.port < 65536
        serve_preview(config, args.host, args.port)
    elif args.command == "export":
        destination = (
            args.destination or config.root / "export" / "open-athena.github.io"
        )
        exported = export_workspace(config, destination)
        print(f"Exported {len(exported)} files under {destination.resolve()}")
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
