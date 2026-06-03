"""ChromBPNet-on-gLM-embeddings variant-effect eval (DART-Eval Task 5 / ARSENAL).

Trains/uses a ChromBPNet-style supervised head over a genomic-LM representation
(or one-hot) and scores caQTL/dsQTL variant effects on *our* train/test splits.
See ``snakemake/evals`` for the QTL datasets and GitHub issue #236 for the
overall design.
"""
