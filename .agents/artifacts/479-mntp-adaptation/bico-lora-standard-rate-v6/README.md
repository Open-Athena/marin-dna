# Standard-rate BICO LoRA result

This bundle contains compact evidence from the completed 1,000-step rank-16 BICO LoRA run at learning rate `5e-5`.
The public run is [W&B s6y7ef4j](https://wandb.ai/gonzalobenegas/marin/runs/s6y7ef4j).
The public evaluation artifact is `gonzalobenegas/marin/dna-exp479-bico-lora-r16-lr5e-5-information-gate:v0` with digest `6a4ca1394d03563ece0d67d543aab3bc`.

`mendelian-paired-comparisons.csv` compares every checkpoint with step 0 on identical odd-autosome/X development rows.
The analysis uses 2,000 seed-0 natural-unit bootstrap replicates within each of eight qualifying Mendelian subsets and combines subset deltas using the registered macro convention.
All paired 95% intervals include zero.
The focused figure shows both absolute AUPRC and paired differences from step 0.

The versioned adapters and optimizer-bearing final checkpoint remain private in `s3://oa-bolinas/issues/479/bico-lora-standard-rate/v1` and were not copied into this repository.
No checkpoint was deleted or uploaded to Hugging Face.
No even-autosome or chromosome-Y labels, predictions, effects, or metrics were accessed.
