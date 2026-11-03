"""Task 3: supervised neural PCFG on Treebank Partition A.

Run from the project folder:
    python src/task3.py --train --device auto
    python src/task3.py --split test --device cpu
    python src/task3.py --split validation --show-tree 1 --limit 3

Training is explicit. Evaluation loads the saved checkpoint and does not train.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import torch
from nltk.corpus import treebank
from nltk.tree import Tree
from torch import nn
from torch.nn import functional as F


SEED = 42
PUNCT_TAGS = {".", ",", ":", "``", "''", "-LRB-", "-RRB-"}
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent if HERE.name == "src" else HERE


def load_partition_a():
    try:
        trees = list(treebank.parsed_sents())
    except LookupError:
        if not nltk.download("treebank", quiet=True):
            raise RuntimeError("Could not download the NLTK treebank corpus")
        trees = list(treebank.parsed_sents())

    if len(trees) != 3914:
        raise ValueError(f"Expected 3914 trees, found {len(trees)}")

    indices = np.arange(len(trees))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    partition_a = np.array(indices[:1305], dtype=int)

    np.random.seed(SEED)
    np.random.shuffle(partition_a)
    train_size = int(0.80 * len(partition_a))
    val_size = int(0.10 * len(partition_a))
    indices_by_split = {
        "train": partition_a[:train_size],
        "validation": partition_a[train_size:train_size + val_size],
        "test": partition_a[train_size + val_size:],
    }

    groups = {
        name: [clean_tree_labels(trees[int(i)]) for i in split_indices]
        for name, split_indices in indices_by_split.items()
    }
    assert [len(groups[name]) for name in ("train", "validation", "test")] == [1044, 130, 131]
    assert all(tree is not None for group in groups.values() for tree in group)
    return groups


def clean_tree_labels(tree):
    def clean(node):
        if node.label() == "-NONE-":
            return None

        children = []
        for child in node:
            if isinstance(child, Tree):
                kept = clean(child)
                if kept is not None:
                    children.append(kept)
            else:
                children.append(child)

        if not children:
            return None

        label = node.label()
        if not label.startswith("-"):
            label = label.split("-", 1)[0].split("=", 1)[0]
        return Tree(label, children)

    return clean(tree)


def replace_rare_words(trees, cutoff=1):
    counts = Counter(word for tree in trees for word in tree.leaves())
    changed_trees = []
    for tree in trees:
        changed = tree.copy(deep=True)
        for position in changed.treepositions("leaves"):
            if counts[changed[position]] <= cutoff:
                changed[position] = "<UNK>"
        changed_trees.append(changed)
    return changed_trees


def to_cnf(tree):
    converted = tree.copy(deep=True)
    converted.collapse_unary(collapsePOS=True, collapseRoot=True)
    converted.chomsky_normal_form(factor="right", horzMarkov=1)
    return converted


def grammar_info(train_trees):
    cnf_trees = [to_cnf(tree) for tree in replace_rare_words(train_trees)]
    binary_labels = set()
    word_labels = set()
    binary_rules = set()

    for tree in cnf_trees:
        for rule in tree.productions():
            parent = rule.lhs().symbol()
            if len(rule.rhs()) == 2:
                left, right = (child.symbol() for child in rule.rhs())
                binary_labels.add(parent)
                binary_rules.add((parent, left, right))
            else:
                word_labels.add(parent)

    nt_labels = sorted(binary_labels)
    pt_labels = sorted(word_labels)
    roots = sorted({tree.label() for tree in cnf_trees})
    pairs = sorted({(left, right) for _, left, right in binary_rules})
    words = {word for tree in cnf_trees for word in tree.leaves()}
    vocab = ["<UNK>"] + sorted(words - {"<UNK>"})

    assert not binary_labels.intersection(word_labels)
    assert len(cnf_trees) == 1044
    return {
        "trees": cnf_trees,
        "nt_labels": nt_labels,
        "pt_labels": pt_labels,
        "binary_rules": sorted(binary_rules),
        "rhs_pairs": pairs,
        "root_labels": roots,
        "vocab": vocab,
        "nt_to_id": {label: i for i, label in enumerate(nt_labels)},
        "pt_to_id": {label: i for i, label in enumerate(pt_labels)},
        "pair_to_id": {pair: i for i, pair in enumerate(pairs)},
        "root_to_id": {label: i for i, label in enumerate(roots)},
        "word_to_id": {word: i for i, word in enumerate(vocab)},
    }


class Task3Grammar(nn.Module):
    def __init__(self, info):
        super().__init__()
        nt_count = len(info["nt_labels"])
        pt_count = len(info["pt_labels"])

        self.nt_embedding = nn.Embedding(nt_count, 256)
        self.pt_embedding = nn.Embedding(pt_count, 256)
        self.root_embedding = nn.Parameter(torch.randn(256) * 0.02)

        def mlp(output_size):
            return nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, output_size),
            )

        self.root_mlp = mlp(len(info["root_labels"]))
        self.rule_mlp = mlp(len(info["rhs_pairs"]))
        self.word_mlp = mlp(len(info["vocab"]))

        allowed = torch.zeros(nt_count, len(info["rhs_pairs"]), dtype=torch.bool)
        for parent, left, right in info["binary_rules"]:
            allowed[info["nt_to_id"][parent], info["pair_to_id"][(left, right)]] = True
        assert allowed.any(dim=1).all()
        self.register_buffer("allowed", allowed)

    def forward(self):
        root_log = F.log_softmax(
            self.root_mlp(self.root_embedding), dim=-1
        )
        rule_scores = self.rule_mlp(self.nt_embedding.weight)
        rule_scores = rule_scores.masked_fill(~self.allowed, float("-inf"))
        binary_log = F.log_softmax(rule_scores, dim=-1)
        word_log = F.log_softmax(
            self.word_mlp(self.pt_embedding.weight), dim=-1
        )
        return root_log, binary_log, word_log


def gold_tree_log_prob(tree, scores, info):
    # For a fully observed gold tree, its inside score is the sum of the
    # log probabilities of the root, binary rules, and lexical emissions.
    root_log, binary_log, word_log = scores

    def inside(node):
        if isinstance(node[0], str):
            return word_log[
                info["pt_to_id"][node.label()],
                info["word_to_id"][node[0]],
            ]

        left, right = node
        return (
            binary_log[
                info["nt_to_id"][node.label()],
                info["pair_to_id"][(left.label(), right.label())],
            ]
            + inside(left)
            + inside(right)
        )

    return root_log[info["root_to_id"][tree.label()]] + inside(tree)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or auto")
    return torch.device(name)


def train_model(model, info, epochs, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                 betas=(0.75, 0.999))
    trees = info["trees"]

    for epoch in range(1, epochs + 1):
        model.train()
        if epoch == 1:
            order = list(range(len(trees)))
            random.Random(SEED).shuffle(order)
        else:
            generator = torch.Generator().manual_seed(SEED + epoch)
            order = torch.randperm(len(trees), generator=generator).tolist()

        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            scores = model()
            loss = torch.stack([
                -gold_tree_log_prob(trees[i], scores, info)
                for i in batch_indices
            ]).mean()
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_indices)

        print(f"Epoch {epoch:02d}: average gold-tree NLL "
              f"{total_loss / len(trees):.3f}", flush=True)

    return optimizer


class CKYDecoder:
    def __init__(self, model, info):
        model.eval()
        self.symbols = info["nt_labels"] + info["pt_labels"]
        self.nt_count = len(info["nt_labels"])
        self.word_ids = info["word_to_id"]
        symbol_ids = {name: i for i, name in enumerate(self.symbols)}

        with torch.no_grad():
            root_log, binary_log, word_log = model()
        self.word_matrix = word_log.cpu().numpy()
        self.root_ids = np.array(
            [symbol_ids[name] for name in info["root_labels"]],
            dtype=np.int32,
        )
        self.root_scores = root_log.cpu().numpy()

        rows = [
            (
                symbol_ids[parent], symbol_ids[left], symbol_ids[right],
                float(binary_log[
                    info["nt_to_id"][parent],
                    info["pair_to_id"][(left, right)],
                ]),
            )
            for parent, left, right in info["binary_rules"]
        ]
        self.parent = np.array([row[0] for row in rows], dtype=np.int32)
        self.left = np.array([row[1] for row in rows], dtype=np.int32)
        self.right = np.array([row[2] for row in rows], dtype=np.int32)
        self.rule_score = np.array([row[3] for row in rows], dtype=np.float32)
        self.rule_ids = np.arange(len(rows))

        self.order = np.argsort(self.parent, kind="stable")
        sorted_parents = self.parent[self.order]
        self.start = np.r_[0, np.flatnonzero(np.diff(sorted_parents)) + 1]
        self.unique_parents = sorted_parents[self.start]
        self.group = np.repeat(
            np.arange(len(self.start)),
            np.diff(np.r_[self.start, len(rows)]),
        )

    def decode(self, words):
        n = len(words)
        if not n:
            return None

        chart = np.full((n, n + 1, len(self.symbols)),
                        -np.inf, dtype=np.float32)
        back_rule = np.full(chart.shape, -1, dtype=np.int32)
        back_split = np.full(chart.shape, -1, dtype=np.int32)

        word_ids = [self.word_ids.get(word, 0) for word in words]
        chart[np.arange(n), np.arange(1, n + 1), self.nt_count:] = (
            self.word_matrix[:, word_ids].T
        )

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                splits = np.arange(i + 1, j)
                candidates = (
                    chart[i, splits[:, None], self.left[None, :]]
                    + chart[splits[:, None], j, self.right[None, :]]
                    + self.rule_score[None, :]
                )
                best_split = candidates.argmax(axis=0)
                rule_scores = candidates[best_split, self.rule_ids]
                sorted_scores = rule_scores[self.order]
                parent_scores = np.maximum.reduceat(sorted_scores, self.start)
                winners = sorted_scores == parent_scores[self.group]
                positions = np.minimum.reduceat(
                    np.where(winners, self.rule_ids, len(self.rule_ids)),
                    self.start,
                )
                best_rule = self.order[positions]
                valid = np.isfinite(parent_scores)
                parents = self.unique_parents[valid]
                rules = best_rule[valid]

                chart[i, j, parents] = parent_scores[valid]
                back_rule[i, j, parents] = rules
                back_split[i, j, parents] = splits[best_split[rules]]

        root_values = chart[0, n, self.root_ids] + self.root_scores
        root_choice = root_values.argmax()
        if not np.isfinite(root_values[root_choice]):
            return None

        def build(i, j, label_id):
            rule_id = back_rule[i, j, label_id]
            if rule_id == -1:
                return Tree(self.symbols[label_id], [words[i]])
            k = back_split[i, j, label_id]
            return Tree(self.symbols[label_id], [
                build(i, k, self.left[rule_id]),
                build(k, j, self.right[rule_id]),
            ])

        return build(0, n, self.root_ids[root_choice])


def raw_brackets(tree):
    spans = set()

    def visit(node, start, is_root=False):
        if isinstance(node, str):
            return start + 1
        end = start
        for child in node:
            end = visit(child, end)
        if not is_root and any(isinstance(child, Tree) for child in node):
            spans.add((start, end))
        return end

    visit(tree, 0, is_root=True)
    return spans


def scored_brackets(tree, gold_tree):
    keep = [tag not in PUNCT_TAGS for _, tag in gold_tree.pos()]
    prefix = [0]
    for keep_word in keep:
        prefix.append(prefix[-1] + int(keep_word))

    spans = set()
    for start, end in raw_brackets(tree):
        left, right = prefix[start], prefix[end]
        if 2 <= right - left < prefix[-1]:
            spans.add((left, right))
    return spans


def evaluate(gold_trees, decoder, show_tree=0):
    sentence_scores = []
    correct = predicted_total = gold_total = 0
    complete = skipped = shown = 0

    for i, gold_tree in enumerate(gold_trees):
        if sum(tag not in PUNCT_TAGS for _, tag in gold_tree.pos()) < 2:
            skipped += 1
            continue

        predicted_cnf = decoder.decode(gold_tree.leaves())
        predicted_tree = None
        predicted = set()
        if predicted_cnf is not None:
            complete += 1
            predicted_tree = predicted_cnf.copy(deep=True)
            predicted_tree.un_chomsky_normal_form()
            assert predicted_tree.leaves() == gold_tree.leaves()
            predicted = scored_brackets(predicted_tree, gold_tree)

        gold = scored_brackets(gold_tree, gold_tree)
        matches = len(predicted & gold)
        correct += matches
        predicted_total += len(predicted)
        gold_total += len(gold)
        total = len(predicted) + len(gold)
        sentence_scores.append(2 * matches / total if total else 1.0)

        if shown < show_tree:
            print(f"\nSentence {i}: {' '.join(gold_tree.leaves())}")
            print("Predicted:", predicted_tree)
            print("Gold:", gold_tree)
            shown += 1
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(gold_trees)}", flush=True)

    return {
        "scored": len(sentence_scores),
        "skipped": skipped,
        "complete": complete,
        "sentence_f1": sum(sentence_scores) / len(sentence_scores),
        "corpus_f1": (
            2 * correct / (predicted_total + gold_total)
            if predicted_total + gold_total else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="train from scratch")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=("validation", "test", "both"),
                        default="test")
    parser.add_argument("--limit", type=int, help="score only the first N trees")
    parser.add_argument("--show-tree", type=int, default=0)
    parser.add_argument("--output", type=Path, help="save metrics as JSON")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.show_tree < 0:
        parser.error("--show-tree cannot be negative")

    device = choose_device(args.device)
    groups = load_partition_a()
    info = grammar_info(groups["train"])
    torch.manual_seed(SEED)
    model = Task3Grammar(info).to(device)
    print(f"Device: {device} | train: {len(groups['train'])} | "
          f"validation: {len(groups['validation'])} | test: {len(groups['test'])}")

    default_path = PROJECT_ROOT / "checkpoints" / "task3_epoch15.pt"
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif (not args.train and not default_path.is_file()
          and (PROJECT_ROOT / "task3_epoch15.pt").is_file()):
        checkpoint_path = PROJECT_ROOT / "task3_epoch15.pt"
    else:
        checkpoint_path = default_path

    if args.train:
        optimizer = train_model(model, info, args.epochs, args.batch_size)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "nt_labels": info["nt_labels"],
            "pt_labels": info["pt_labels"],
            "rhs_pairs": info["rhs_pairs"],
            "root_labels": info["root_labels"],
            "vocab": info["vocab"],
        }, checkpoint_path)
        print("Saved checkpoint:", checkpoint_path)
    else:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Use --train or pass --checkpoint PATH."
            )
        saved = torch.load(checkpoint_path, map_location=device,
                           weights_only=True)
        for key in ("nt_labels", "pt_labels", "rhs_pairs", "root_labels", "vocab"):
            if key in saved and saved[key] != info[key]:
                raise ValueError(f"Checkpoint has a different {key} ordering")
        model.load_state_dict(saved["model_state"])
        print("Loaded checkpoint:", checkpoint_path,
              "epoch:", saved.get("epoch", "unknown"))

    decoder = CKYDecoder(model, info)
    splits = ("validation", "test") if args.split == "both" else (args.split,)
    results = {}
    for split in splits:
        gold_trees = groups[split]
        if args.limit is not None:
            gold_trees = gold_trees[:args.limit]
        result = evaluate(gold_trees, decoder, args.show_tree)
        results[split] = result
        print(f"{split}: scored {result['scored']}/{len(gold_trees)} | "
              f"complete {result['complete']} | "
              f"sentence F1 {result['sentence_f1']:.3f} | "
              f"corpus F1 {result['corpus_f1']:.3f}")

    if args.limit is not None:
        print("Subset check: run without --limit for final results.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("Saved results:", args.output)


if __name__ == "__main__":
    main()
