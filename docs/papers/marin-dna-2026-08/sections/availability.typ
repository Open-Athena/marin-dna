= Data, Code, Models, and Live Resources
<availability>
The #link("https://github.com/Open-Athena/marin-dna")[MarinDNA repository] contains the data pipelines, evaluation implementation, manuscript source, and tracked investigation scripts.
The mechanical manuscript and frozen figure baseline is available at commit #link("https://github.com/Open-Athena/marin-dna/tree/3b608d39b41c2330636ec647dbb25d26b0895187/docs/papers/marin-dna-2026-08")[3b608d39b41c2330636ec647dbb25d26b0895187].
The final preprint revision will identify commit-pinned training, dataset-construction, evaluation, and plotting code for each retained result.

The frozen Mendelian benchmark is #link("https://huggingface.co/datasets/bolinas-dna/evals_mendelian_traits/tree/4aed58e50c5dea0b878a665007af2ef9e5108e9f")[bolinas-dna/evals_mendelian_traits at revision 4aed58e].
The frozen saturation-genome-editing benchmark is #link("https://huggingface.co/datasets/bolinas-dna/evals_sge/tree/225d3d1ea32a4af547891b13c33b5e92a5aae849")[bolinas-dna/evals_sge at revision 225d3d1].
Training datasets, evaluation datasets, and the released model are grouped in the #link("https://huggingface.co/collections/marin-dna/a-1b-standard-transformer-rivals-evo-2-40b-on-vep")[MarinDNA Hugging Face collection].
The exact public model and training-dataset revisions will be pinned here before the preprint is released.

The #link("https://openathena.ai/marin-dna/")[MarinDNA leaderboard] is an evolving project resource.
Its contents may change as models and baselines are added and must not be treated as the frozen comparison reported in this manuscript.
The committed manuscript figures and their recorded input revisions define the archival result snapshot.

Optional interactive resources include:

- #link("https://molab.marimo.io/notebooks/nb_MrPpr5xYcN3HGt5tLY86bk/app")[Interactive sequence explorer].
- #link("https://molab.marimo.io/github/Open-Athena/marin-dna/blob/5d1925fe0d6569c0ee0c29db06b8f287c2347065/examples/model_inference_and_vep.py")[Model inference and BRCA1 variant-effect prediction notebook].
