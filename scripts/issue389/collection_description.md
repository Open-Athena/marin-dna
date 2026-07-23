Resources released and curated for the [Genomic Language Model Optimization](https://openathena.ai/blog/genomic-lm-optimization/) blog post.

- **Released model:** MarinDNA m5.1 is the final base-model checkpoint released with the blog.
- **Training datasets:** CDS, upstream, downstream, enhancer, and ncRNA sequence resources.
- **Training-validation probes:** five matched validation datasets used to monitor the corresponding training regions; these probes were not training data.
- **Downstream evaluation only:** Mendelian variant effects and saturation genome editing (SGE); neither dataset was training data.

The five training resources describe the data inventory used across the m5.1 lineage; they did not all contribute to every training phase. Mendelian and SGE use different construction and aggregation protocols, so their score levels are not directly comparable.
