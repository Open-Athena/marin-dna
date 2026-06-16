"""Low-constraint sites: |phyloP| < threshold AND phastCons == 0.

Heavy I/O — scans the two ~5-9 GB bigWigs window-by-window. Single-threaded
(pyBigWig); no conda (uses the pipeline's own python env).
"""


rule conserved_sites:
    input:
        phylop="results/downloads/phylop.bw",
        phastcons="results/downloads/phastcons.bw",
    output:
        "results/conserved/conserved_sites.bed",
    run:
        df = scan_neutral_intervals(
            input.phylop,
            input.phastcons,
            CHROMS,
            config["phylop_threshold"],
            window_size=config["window_size"],
        )
        assert len(df) > 0, "no low-constraint sites found — check tracks/threshold"
        df.to_csv(output[0], sep="\t", header=False, index=False)
        print(f"[neutral_sites] {len(df):,} low-constraint intervals")
