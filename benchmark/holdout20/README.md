# Fixed manuscript 20% held-out test set

This directory contains 332 held-out samples from the 1,656-sample cohort, spanning 21 molecular subtypes. The samples and labels agree across all three supplied manuscript prediction files.

- `expression_log2TPM1.csv.gz`: genes × samples; first column `gene_id`, remaining columns sample IDs. Values are already log2(TPM+1).
- `labels.csv`: sample IDs and true subtype (`category`).
- `sample_metadata.csv`: dataset of origin and subtype.
- `split.csv`: fixed train/test assignment for all 1,656 included samples. The training partition is the complement of the 332 test IDs.
- `subtype_counts.csv`: held-out sample counts.
- `provenance.json`: source and output checksums.

Evaluate a compatible packaged checkpoint:

```bash
python -m attentionaml.predict --checkpoint checkpoints/AttentionAML.pt --expression benchmark/holdout20/expression_log2TPM1.csv.gz --output results/holdout_predictions.csv
```

Checkpoint gene IDs/order and preprocessing must match. To retrain using the full expression dataset and full labels, provide `--split-file benchmark/holdout20/split.csv` to AML_holdout.py. The full training expression data are not included here.
