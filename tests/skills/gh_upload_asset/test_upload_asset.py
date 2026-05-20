"""Tests for pure helpers in the gh-upload-asset skill script.

The script lives under ``.agents/skills/`` rather than ``src/marin_dna/`` (it's a
skill-bundled tool, not part of the library), so we load it by file path.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2].parent
SCRIPT_PATH = (
    REPO_ROOT / ".agents" / "skills" / "gh-upload-asset" / "scripts" / "upload_asset.py"
)

_spec = importlib.util.spec_from_file_location("upload_asset", SCRIPT_PATH)
upload_asset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upload_asset)


class TestFormatOutput:
    URL = "https://gist.githubusercontent.com/u/abc/raw/sha/plot.png"

    def test_auto_image_extension_renders_markdown(self):
        assert (
            upload_asset.format_output("plot.png", self.URL)
            == f"![plot.png]({self.URL})"
        )

    def test_auto_non_image_extension_renders_plain_url(self):
        assert upload_asset.format_output("data.csv", self.URL) == self.URL

    def test_auto_image_mime_without_known_extension_renders_markdown(self):
        result = upload_asset.format_output("plot.tiff", self.URL)
        assert result == f"![plot.tiff]({self.URL})"

    def test_explicit_url_format_overrides_auto(self):
        assert (
            upload_asset.format_output("plot.png", self.URL, format_type="url")
            == self.URL
        )

    def test_explicit_img_format_emits_html_tag(self):
        result = upload_asset.format_output("plot.png", self.URL, format_type="img")
        assert result == f'<img alt="plot.png" src="{self.URL}" />'

    def test_custom_alt_text_is_used(self):
        result = upload_asset.format_output("plot.png", self.URL, alt_text="my plot")
        assert result == f"![my plot]({self.URL})"

    def test_extension_matching_is_case_insensitive(self):
        assert (
            upload_asset.format_output("plot.PNG", self.URL)
            == f"![plot.PNG]({self.URL})"
        )


class TestUniqueName:
    def test_returns_basename_when_unique(self):
        paths = ["a/plot.png", "b/chart.png"]
        assert upload_asset._unique_name("a/plot.png", paths) == "plot.png"

    def test_prefixes_parent_dir_when_basenames_collide(self):
        paths = ["run1/plot.png", "run2/plot.png"]
        assert upload_asset._unique_name("run1/plot.png", paths) == "run1_plot.png"
        assert upload_asset._unique_name("run2/plot.png", paths) == "run2_plot.png"

    def test_walks_up_until_name_becomes_unique(self):
        paths = ["exp_a/run1/plot.png", "exp_b/run1/plot.png"]
        assert (
            upload_asset._unique_name("exp_a/run1/plot.png", paths)
            == "exp_a_run1_plot.png"
        )

    def test_falls_back_to_basename_when_no_ancestor_disambiguates(self):
        paths = ["plot.png", "plot.png"]
        assert upload_asset._unique_name("plot.png", paths) == "plot.png"
