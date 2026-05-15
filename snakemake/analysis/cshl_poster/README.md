# CSHL 2026 poster

Source for the conference poster Gonzalo presents at **CSHL 90th Symposium: AI in Biology** (May 26–31, 2026). Title: *Data curation strategies for genomic language models* (Gonzalo Benegas, Eric Czech — Open Athena).

The poster is a single self-contained HTML page (`poster.html`) sized as a 44″ × 44″ square. Open it in a browser to preview; export to PDF manually for submission.

## Deadlines

| | Date | Notes |
| --- | --- | --- |
| **CSHL print service** | **Tue May 19, 2026** | `meetings.cshl.edu/posterprservice.aspx` — $110 standard; late fees $10–25. One day **before** the virtual deadline. |
| **Virtual poster PDF** | **Wed May 20, 2026** | `meetings.cshl.edu/submitposter/` — uploads to the Virtual Poster Gallery. |

Cross-reference the `Poster FAQ.pdf` attached to the May 13 "IMPORTANT INFORMATION" email from `pakaluk@cshl.edu` for any meeting-specific overrides (max upload size, allowed PDF version, etc.).

## How to view

```bash
# From the repo root:
python3 -m http.server --directory snakemake/analysis/cshl_poster 8765
# then open http://localhost:8765/poster.html in Brave or Chrome.
```

The page uses `transform: scale(0.25)` under `@media screen` so a 44″ poster fits inside a normal browser window. The print stylesheet is unaffected.

## How to export the PDF

The poster repo does not script PDF rendering — do it manually:

1. Open `poster.html` in Brave (or Chrome).
2. `Cmd+P` → "Save as PDF".
3. In the Print dialog, set **Paper size → Custom → 44 in × 44 in** and **Margins → None**. Disable headers/footers.
4. Save outside the repo (PDF artifacts are not committed).

## File layout

```
poster.html              # the poster (HTML + inline CSS)
figs/
  r1.svg / r2.svg / t1.svg / t2.svg     # result-panel figures
  chromosome_schematic.svg              # left-column schematic
  phylo_schematic.svg                   # right-column schematic
  qr-repo.svg / qr-contact.svg          # QR codes
  icons/
    oa-logo.svg                         # OA lockup (text + icon) from openathena.ai
fonts/
  Herbik-Regular.ttf                    # OA heading font (self-hosted)
README.md
```

## Source mapping

Every headline number on the poster should trace back to a notebook / pipeline / issue. Until real figures replace the placeholder SVGs, this table is the working list.

| Figure / number | Status | Source |
| --- | --- | --- |
| Pipeline schematic (data → Qwen → eval) | drawn from scratch in SVG | — |
| Headline `0.77 vs 0.57` AUPRC (badge inside R1) | from abstract | TraitGym promoter VEP, 6M region-specific vs Evo 2 40B — TODO: pin to specific `scripts/evo2_eval/traitgym_v2_metrics.py` parquet output + commit. |
| R1 (region-specific beats 40B) | placeholder | TODO |
| R2 (balanced sampling) | placeholder | TODO |
| T1 (mammals optimal for promoters) | placeholder | TODO |
| T2 (animals optimal for CDS) | placeholder | TODO |
| Functional regions + evolutionary timescales illustrations | embedded in the DATA card of `pipeline_schematic.svg` (single place, not duplicated in body) | — |

## Brand assets — pinned commits

The poster mirrors the openathena.ai website brand. The website is the
single source of truth; everything below is vendored from the same
commit so re-fetches reproduce exactly.

Source repo: `Open-Athena/open-athena.github.io@ba5e9bfc8287896cdfb74952388707ec082b3ea4`.

- **Palette** — CSS custom properties (`--bg`, `--accent`, `--font-heading`, …) copied verbatim from `/static/css/style.css`.
- **Plotly colorway** — the 8-colour data-viz palette taken from the inline Plotly defaults block of `/blog/delphi/` (the closest peer artifact to a poster).
- **Herbik-Regular.ttf** (heading font) — vendored from `/static/assets/fonts/Herbik-Regular.ttf`. © 2024 Daniel Veneklaas (vendored under the same terms as the website).
- **OA logo lockup** (`oa-logo.svg`) — vendored from `/static/assets/images/logo.svg`, the same asset the website uses in its nav bar.
- **Lato** loaded from Google Fonts at view time.
- **Authoring voice** follows `/.claude/skills/blog-post/SKILL.md` — sober numbers over adjectives, results-first sentences, no AI-explainer cadence.

## Things still to wire up before submission

- Replace `figs/r1.svg`–`t2.svg` with real result plots (kept in the same OA palette via `OA_COLORWAY` once figure-restyling lands).
- Confirm authors + affiliations match the submitted abstract verbatim.
- Decide whether to add a bioRxiv QR; current poster intentionally omits (no preprint yet).
- Test both QR codes scan via phone camera at print size.
- Pre-submission: open `Poster FAQ.pdf` for any format/size constraints.
