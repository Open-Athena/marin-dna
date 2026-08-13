= Open Development and Provenance
<provenance>
MarinDNA was developed through public, issue-tracked experiments in which hypotheses, configurations, intermediate results, and negative findings were recorded as the work progressed.
The issue history provides development provenance but is not required scientific exposition: the manuscript states the methods, evidence, and limitations needed to interpret its claims.
Decisions and substantial editorial changes for this frozen preprint are tracked in #link("https://github.com/Open-Athena/marin-dna/issues/449")[issue \#449].
The mechanical manuscript baseline and its 20 SVG assets were fixed at MarinDNA commit #link("https://github.com/Open-Athena/marin-dna/tree/3b608d39b41c2330636ec647dbb25d26b0895187/docs/papers/marin-dna-2026-08")[3b608d39b41c2330636ec647dbb25d26b0895187], whose content and figures were converted from project commit d8c4803cbbbffafb24890cd0c75134d78368d55c.
No new training run or evaluation dataset was introduced during that conversion.

Key result families retain the following development records, which will be expanded with exact comments, code commits, checkpoints, dataset revisions, and figure inputs in the supplementary provenance table:

- Data specialists and mixture balance: #link("https://github.com/Open-Athena/marin-dna/issues/21")[\#21], #link("https://github.com/Open-Athena/marin-dna/issues/27")[\#27], and #link("https://github.com/Open-Athena/marin-dna/issues/13")[\#13].
- Manual scaling failure and the need for transfer: #link("https://github.com/Open-Athena/marin-dna/issues/57")[\#57].
- Validation-loss design and interpretation: #link("https://github.com/Open-Athena/marin-dna/issues/8")[\#8].
- Frozen-embedding probe development: #link("https://github.com/Open-Athena/marin-dna/issues/369")[\#369] and the evaluation-pipeline records it references.
- Frozen MarinDNA–Evo 2 comparison: the #link("https://github.com/Open-Athena/marin-dna/issues/131#issuecomment-4498438127")[refreshed Evo 2 score record] and the #link("https://github.com/Open-Athena/marin-dna/blob/e9f32d582b56dba4a140f71227d244bb6435fed1/scripts/issue449_audit_headline.py")[commit-pinned paired analysis].
- Training compute and throughput: #link("https://github.com/Open-Athena/marin-dna/issues/135")[\#135], #link("https://github.com/Open-Athena/marin-dna/issues/354")[\#354], #link("https://github.com/Open-Athena/marin-dna/issues/430")[\#430], and the #link("https://github.com/Open-Athena/marin-dna/blob/060e3bdb49f90c831c7bd778da55092d320bbe45/scripts/issue449_audit_efficiency.py")[commit-pinned arithmetic audit].

Issue links are provided to make the branching research history inspectable; scholarly claims about prior work use bibliographic citations, and claims about MarinDNA are supported by the manuscript's figures, Methods, and frozen artifacts.
