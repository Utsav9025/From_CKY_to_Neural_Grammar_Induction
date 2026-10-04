import nltk
from nltk.corpus import treebank
import numpy as np

def load_all_trees():
    nltk.download("treebank", quiet=True)
    trees = list(treebank.parsed_sents())
    assert len(trees) == 3914, f"Expected 3914 trees, found {len(trees)}"
    return trees


def get_partition_a_indices(trees):
    indices = np.arange(len(trees))
    np.random.seed(42)
    np.random.shuffle(indices)
    return indices[0:1305].tolist()  

def split_partition_a(partition_a):
    shuffled = np.array(partition_a, dtype=int)

    # Start a new, independent shuffle within Partition A
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
    print(f"Loaded {len(trees)} trees")

    partition_a = get_partition_a_indices(trees)
    print(f"Partition A: {len(partition_a)} trees")
    # print(f"First assigned tree:\n{trees[partition_a[0]]}") 
    splits = split_partition_a(partition_a)

    for name, indices in splits.items():
        print(f"{name}: {len(indices)} trees")