import json
from pathlib import Path

import nltk
import numpy as np
from nltk.corpus import treebank


def load_all_trees():
    nltk.download("treebank", quiet=True)
    trees = list(treebank.parsed_sents())
    assert len(trees) == 3914, f"Expected 3914 trees, found {len(trees)}"
    return trees


def get_partitions(trees):
    indices = np.arange(len(trees))
    np.random.seed(42)
    np.random.shuffle(indices)

    return {
        "A": indices[:1305].tolist(),
        "B": indices[1305:2610].tolist(),
        "C": indices[2610:3914].tolist(),
    }


def split_partition_a(partition_a):
    shuffled = np.array(partition_a, dtype=int)

    # A separate shuffle for the train, validation, and test split
    np.random.seed(42)
    np.random.shuffle(shuffled)

    n_train = int(0.80 * len(shuffled))
    n_val = int(0.10 * len(shuffled))

    return {
        "train": shuffled[:n_train].tolist(),
        "validation": shuffled[n_train:n_train + n_val].tolist(),
        "test": shuffled[n_train + n_val:].tolist(),
    }


if __name__ == "__main__":
    trees = load_all_trees()
    partitions = get_partitions(trees)
    splits = split_partition_a(partitions["A"])

    print(f"Loaded {len(trees)} trees")

    for name, indices in partitions.items():
        print(f"Partition {name}: {len(indices)} trees")

    for name, indices in splits.items():
        print(f"Partition A {name}: {len(indices)} trees")

    output = Path(__file__).resolve().parent.parent / "data" / "partition_indices.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    saved_indices = {
        "partitions": partitions,
        "partition_a_splits": splits,
    }
    output.write_text(json.dumps(saved_indices, indent=2), encoding="utf-8")
    print(f"Saved indices to {output}")