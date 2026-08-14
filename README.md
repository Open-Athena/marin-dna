<h1 align="center">MarinDNA</h1>

<p align="center">Open development of genomic language models — data, modeling, and evaluation.</p>

<p align="center"><sub>Inspired by <a href="https://github.com/marin-community/marin">Marin</a>.</sub></p>

## News

- **2026-08-03** — *Blog post* — [A 1B standard Transformer rivals Evo 2 40B on variant effect prediction](https://openathena.ai/blog/marin-dna/).
- **2026-05-26** — *Poster* — [Data curation strategies for genomic language models](https://github.com/Open-Athena/marin-dna/blob/190999b89b573b07026eff58c2c0a8cc8fa74458/docs/posters/cshl26/poster.pdf) at the [CSHL 90th Symposium "AI in Biology"](https://meetings.cshl.edu/meetings.aspx?meet=SYMP&year=26).

## Research questions

These documents synthesize MarinDNA's current answers and help organize future experiments. Current priorities are an unordered, human-set subset.

### Current priorities

- [Can autoregressive RAG gLMs be accurate and practical?](docs/research/questions/retrieval-augmented-models.md)
- [Can causal gLMs become bidirectional representation and arbitrary-order generation models?](docs/research/questions/bidirectional-models.md)
- [How should genomic anchors be selected and projected across species?](docs/research/questions/genomic-anchors.md)
- [Why do MarinDNA models lag on complex-trait VEP?](docs/research/questions/complex-trait-vep.md)

### Other active questions

- [Can gLM pretraining improve human sequence-to-function modeling?](docs/research/questions/sequence-to-function.md)
- [Does conditioning on species/clade help?](docs/research/questions/species-conditioning.md)
- [How should a short-context gLM acquire long-range context?](docs/research/questions/long-context.md)
- [How should evolutionary timescale shape training?](docs/research/questions/evolutionary-timescale.md)
- [How to optimize pretraining data mixtures?](docs/research/questions/data-mixtures.md)
- [Tokenization](docs/research/questions/tokenization.md)
- [What latent biological features do gLMs learn?](docs/research/questions/latent-features.md)
- [Which genomic regions to train on, and how to find them?](docs/research/questions/training-regions.md)

Bounded experiments remain [GitHub issues](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Aexperiment).

## Resources

- [🤗 Models and datasets on Hugging Face](https://huggingface.co/marin-dna)
- [🏆 Variant effect prediction leaderboard](https://openathena.ai/marin-dna/)
- [🧬 Interactive sequence explorer](https://molab.marimo.io/notebooks/nb_MrPpr5xYcN3HGt5tLY86bk/app)
- [💻 Model inference and BRCA1 variant effect prediction notebook](https://molab.marimo.io/github/Open-Athena/marin-dna/blob/5d1925fe0d6569c0ee0c29db06b8f287c2347065/examples/model_inference_and_vep.py)

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
