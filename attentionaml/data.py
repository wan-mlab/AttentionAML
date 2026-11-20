"""Strict sample alignment and input checks; no learned preprocessing."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_expression(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as stream:
        header = next(csv.reader(stream))
    samples = [v for v in header if v and v != 'gene_id' and not v.startswith('Unnamed:')]
    if len(samples) != len(set(samples)):
        raise ValueError('Duplicate sample columns in expression CSV.')
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[c for c in frame if c.startswith('Unnamed:')])
    if 'gene_id' not in frame:
        raise ValueError('Expression requires a gene_id column and sample columns.')
    if frame.gene_id.isna().any() or frame.gene_id.duplicated().any():
        raise ValueError('Gene identifiers must be nonmissing and unique.')
    frame = frame.set_index('gene_id')
    genes = frame.index.astype(str).tolist()
    sample_ids = frame.columns.astype(str).tolist()
    values = frame.to_numpy(dtype=np.float32).T.copy()
    if not genes or not sample_ids or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected nonempty, finite, nonnegative log2(TPM+1) expression.')
    return values, sample_ids, genes


def load_data(expression, labels, exclusions):
    X, sample_ids, genes = load_expression(expression)
    table = pd.read_csv(labels, dtype=str)
    if not {'sample_id','category'} <= set(table):
        raise ValueError('Labels require sample_id,category.')
    if table[['sample_id','category']].isna().any().any() or table.sample_id.duplicated().any():
        raise ValueError('Missing labels or duplicated sample IDs.')
    table = table.set_index('sample_id')
    missing = set(sample_ids) - set(table.index)
    if missing:
        raise ValueError(f'Missing labels for {len(missing)} expression samples.')
    categories = table.loc[sample_ids, 'category'].to_numpy()
    keep = ~np.isin(categories, exclusions)
    samples = np.asarray(sample_ids)[keep].tolist()
    classes, y = np.unique(categories[keep], return_inverse=True)
    if len(classes) < 2:
        raise ValueError('At least two included subtypes are required.')
    return X[keep], y.astype(np.int64), samples, genes, classes.tolist()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
