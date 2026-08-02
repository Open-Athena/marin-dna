# exp421/422 complete-family statistical revisits

This small CPU-only environment orchestrates the approved grouped AlphaGenome
L2 scan from #421 and the 35-class broad consequence scan from #422. Scientific
logic remains in each issue's experiment directory; `sky.yaml` only stages the
frozen S3 inputs, runs both analyses sequentially on one EC2 CPU instance, and
uploads each hash manifest last.

The #421 analysis must reuse the accepted biosample mapping and the #436
blocks-1/10/19 25M Mendelian focal archive. The #422 analysis must reuse the
frozen chr21 panel and the joint multi-layer extraction. Neither analysis uses
outcome-aware feature selection.
