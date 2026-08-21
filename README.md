<h1 align="center">MarinDNA</h1>

<p align="center">Open development of genomic language models — data, modeling, and evaluation.</p>

<p align="center"><sub>Inspired by <a href="https://github.com/marin-community/marin">Marin</a>.</sub></p>

## News

- **2026-08-03** — *Blog post* — [A 1B standard Transformer rivals Evo 2 40B on variant effect prediction](https://openathena.ai/blog/marin-dna/).
- **2026-05-26** — *Poster* — [Data curation strategies for genomic language models](https://github.com/Open-Athena/marin-dna/blob/190999b89b573b07026eff58c2c0a8cc8fa74458/docs/posters/cshl26/poster.pdf).

## Research

These documents synthesize MarinDNA's current answers and help organize future experiments.

### Current priorities

- [Bidirectionality](docs/research/questions/bidirectional-models.md)
- [Complex-trait VEP](docs/research/questions/complex-trait-vep.md)
- [Genomic anchor projection](docs/research/questions/genomic-anchors.md)
- [RAG](docs/research/questions/retrieval-augmented-models.md)

### Other active questions

- [Context size](docs/research/questions/long-context.md)
- [Data mixing](docs/research/questions/data-mixtures.md)
- [Evolutionary timescales](docs/research/questions/evolutionary-timescale.md)
- [Latent biological features](docs/research/questions/latent-features.md)
- [Sequence-to-function modeling](docs/research/questions/sequence-to-function.md)
- [Species conditioning](docs/research/questions/species-conditioning.md)
- [Tokenization](docs/research/questions/tokenization.md)
- [Training regions](docs/research/questions/training-regions.md)

## Resources

- [Models and datasets on Hugging Face](https://huggingface.co/marin-dna)
- [Variant effect prediction leaderboard](https://openathena.ai/marin-dna/)
- [Interactive sequence explorer](https://molab.marimo.io/notebooks/nb_MrPpr5xYcN3HGt5tLY86bk/app)
- [Model inference and BRCA1 variant effect prediction notebook](https://molab.marimo.io/github/Open-Athena/marin-dna/blob/5d1925fe0d6569c0ee0c29db06b8f287c2347065/examples/model_inference_and_vep.py)

## Community

Join the [Marin Discord](https://discord.gg/J9CTk7pqcM); MarinDNA discussion happens in the `#dna` channel.

## Citation

If you find datasets, models, or experiments from this repo useful, please cite:

> MarinDNA: open development of genomic language models. Open Athena, 2026.
> https://github.com/Open-Athena/marin-dna

BibTeX:

```bibtex
@misc{marin-dna,
  title  = {MarinDNA: open development of genomic language models},
  author = {{Open Athena}},
  year   = {2026},
  url    = {https://github.com/Open-Athena/marin-dna},
}
```
