from __future__ import annotations

from collections import defaultdict

import numpy as np


def case_grouped_stratified_split(provider, indices, train_frac=0.7, val_frac=0.15, seed=42):
    """Split by (cohort, case_id) groups so no case straddles train/val/test."""
    rng = np.random.default_rng(seed)

    groups: dict[tuple, list[int]] = defaultdict(list)
    group_label: dict[tuple, int] = {}
    for idx in indices:
        rec = provider.get_record(idx)
        key = (rec.cohort, rec.case_id)
        groups[key].append(idx)
        group_label[key] = rec.label

    pos_keys = [k for k, label in group_label.items() if label == 1]
    neg_keys = [k for k, label in group_label.items() if label == 0]

    rng.shuffle(pos_keys)
    rng.shuffle(neg_keys)

    def split_keys(keys):
        n = len(keys)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        return keys[:n_train], keys[n_train:n_train + n_val], keys[n_train + n_val:]

    def flatten(keys):
        out = []
        for k in keys:
            out.extend(groups[k])
        return out

    pos_tr, pos_val, pos_te = split_keys(pos_keys)
    neg_tr, neg_val, neg_te = split_keys(neg_keys)

    train_idx = flatten(pos_tr + neg_tr)
    val_idx   = flatten(pos_val + neg_val)
    test_idx  = flatten(pos_te + neg_te)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    print(f"  Cases total — pos: {len(pos_keys)}, neg: {len(neg_keys)}")
    print(
        "  Cases split — "
        f"train: {len(pos_tr)} pos / {len(neg_tr)} neg, "
        f"val: {len(pos_val)} pos / {len(neg_val)} neg, "
        f"test: {len(pos_te)} pos / {len(neg_te)} neg"
    )
    print(
        "  Slides split — "
        f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )
    return train_idx, val_idx, test_idx
