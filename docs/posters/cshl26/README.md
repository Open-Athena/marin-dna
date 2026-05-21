# CSHL 2026 poster

Source for the conference poster Gonzalo presents at **CSHL 90th Symposium: AI in Biology** (May 26–31, 2026). Title: *Data curation strategies for genomic language models* (Gonzalo Benegas, Eric Czech — Open Athena).

Single-page 44″ × 44″ square poster, white background with navy accent bars (Gemini-style academic layout). Written in [Typst](https://typst.app/) using the `peace-of-posters` template.

## Deadlines

| | Date | Notes |
| --- | --- | --- |
| **CSHL print service** | **Tue May 19, 2026** | `meetings.cshl.edu/posterprservice.aspx` — $110 standard; late fees $10–25. One day **before** the virtual deadline. |
| **Virtual poster PDF** | **Wed May 20, 2026** | `meetings.cshl.edu/submitposter/` — uploads to the Virtual Poster Gallery. |

Cross-reference the `Poster FAQ.pdf` attached to the May 13 "IMPORTANT INFORMATION" email from `pakaluk@cshl.edu` for any meeting-specific overrides (max upload size, allowed PDF version, etc.).

## How to compile

```bash
# Prerequisites (one-time):
brew install typst                 # the compiler
brew install --cask font-inter     # the body / heading font

# From this directory:
typst compile poster.typ           # one-shot → poster.pdf
typst watch poster.typ             # live recompile on every save
```

`poster.pdf` is a build artifact (gitignored). The plots inside `figs/` are regenerated separately by `plots/cshl26_poster.py` at the repo root:

```bash
uv run python plots/cshl26_poster.py
```

## File layout

```
poster.typ              # the poster source (Typst, peace-of-posters template)
poster.pdf              # build artifact — gitignored
.gitignore              # ignores poster.pdf
README.md               # this file
figs/
  region_legend.svg     # gene cartoon — colour key for R1/R2
  timescale_legend.svg  # phylo tree — colour key for T1/T2
  r2_composition.svg    # mixture composition schematic above the R2 line plot
  specialist_bars.svg   # R1 — 3 specialists vs 2 generalists
  r3.svg                # R2 — exp13 mixture sweep over training step
  t1.svg                # T1 — promoter AUPRC vs evolutionary timescale
  t2.svg                # T2 — CDS AUPRC vs evolutionary timescale
  icons/
    oa-logo.svg         # Open Athena lockup (top-right of header)
```

## Source mapping

Every headline number on the poster should trace back to a pipeline / issue.

| Figure | Source |
| --- | --- |
| R1 (specialist vs generalist) | `plot_specialist_grouped_bars` in `plots/cshl26_poster.py`; reads `s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/{exp21-promoters-yolo-step-22000,exp27-cds-yolo-step-34000,exp136-proj_v30-step-9999}/mendelian_traits.parquet` plus Evo 2 40B + GPN-Star M baseline gists. Issues [#27](https://github.com/Open-Athena/marin-dna/issues/27), [#21](https://github.com/Open-Athena/marin-dna/issues/21), [#136](https://github.com/Open-Athena/marin-dna/issues/136). |
| R2 (mixture sweep) | `plot_r3` in `plots/cshl26_poster.py`; reads exp13-{equal,proportional} + exp21 + exp27 step parquets from S3. Issue [#13](https://github.com/Open-Athena/marin-dna/issues/13). |
| T1 (promoter timescales) | `plot_t1` in `plots/cshl26_poster.py`; reads exp55-{humans,primates,mammals,vertebrates,animals} step parquets from S3. Issue [#55](https://github.com/Open-Athena/marin-dna/issues/55). |
| T2 (CDS timescales) | `plot_t2` in `plots/cshl26_poster.py`; reads exp58-{mammals,vertebrates,animals} step parquets from S3. Issue [#58](https://github.com/Open-Athena/marin-dna/issues/58). |
| `region_legend.svg`, `timescale_legend.svg`, `r2_composition.svg` | Hand-drawn SVGs maintained in `figs/`. |
| OA logo | `figs/icons/oa-logo.svg` — vendored from the openathena.ai website. |

## Design choices

- **White background, navy `#1d3557` section title bars.** Departs from the OA brand palette (warm taupe + copper) — the OA palette is comfortable for long-form blog reading but lower-contrast than white-on-white for scan-from-distance viewing. Plots fuse directly with the page background (no frame chrome).
- **Inter** sans-serif throughout (Google Fonts, system-installed via `brew install --cask font-inter`).
- **Plot palettes:**
  - Functional regions (discrete): seaborn `colorblind` — promoter `#0173b2`, CDS `#de8f05`, enhancer `#029e73`.
  - Generalist baselines: neutral greys (Evo 2 40B `#999999`, GPN-Star M `#333333`).
  - Evolutionary timescales (ordinal): `viridis` 5-stop.
  - Mixture proportions (ordinal): `magma` 4-stop.

## To-do before submission

- Confirm authors + affiliations match the submitted abstract verbatim.
- Decide whether to add a bioRxiv link / QR — current poster intentionally omits (no preprint yet).
- Open `Poster FAQ.pdf` for any meeting-specific format constraints.
