"""Train or evaluate the Task 2 Compound PCFG on Partition A.

This uses the architecture, preprocessing, and checkpoint format of the
Task 2 notebook. Training is opt-in; loading a checkpoint does not train.

Examples (run with your virtual environment's Python):
    python src/task2.py --device cpu --split test --decoder mbr
    python src/task2.py --checkpoint checkpoints/task2_best.pt --limit 3
    python src/task2.py --text "The dog runs ."
    python src/task2.py --train --batch-size 8 --epochs 15 --device auto

Dependencies: numpy, nltk, torch, and torch_struct (pytorch-struct).
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import treebank
from nltk.tree import Tree
from torch_struct import SentCFG


PUNCT_TAGS = {".", ",", ":", "``", "''", "-LRB-", "-RRB-"}


def load_all_trees():
    try:
        trees = list(treebank.parsed_sents())
    except LookupError:
        if not nltk.download("treebank", quiet=True):
            raise RuntimeError("Could not download NLTK's treebank corpus.")
        trees = list(treebank.parsed_sents())

    assert len(trees) == 3914, f"Expected 3914 trees, got {len(trees)}"
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
    np.random.seed(42)
    np.random.shuffle(shuffled)

    n_train = int(0.80 * len(shuffled))
    n_val = int(0.10 * len(shuffled))
    return {
        "train": shuffled[:n_train].tolist(),
        "validation": shuffled[n_train:n_train + n_val].tolist(),
        "test": shuffled[n_train + n_val:].tolist(),
    }


def clean_tree_labels(tree):
    def clean(node):
        if node.label() == "-NONE-":
            return None

        children = []
        for child in node:
            if isinstance(child, Tree):
                cleaned_child = clean(child)
                if cleaned_child is not None:
                    children.append(cleaned_child)
            else:
                children.append(child)

        if not children:
            return None

        label = node.label()
        if not label.startswith("-"):
            label = label.split("-", 1)[0].split("=", 1)[0]
        return Tree(label, children)

    return clean(tree)


def load_partition_a():
    trees = load_all_trees()
    splits = split_partition_a(get_partitions(trees)["A"])
    result = {
        name: [clean_tree_labels(trees[i]) for i in indices]
        for name, indices in splits.items()
    }
    assert [len(result[name]) for name in ("train", "validation", "test")] == [1044, 130, 131]
    assert all(t is not None for group in result.values() for t in group)
    return result


def build_vocab(train_sentences):
    counts = Counter(word for sentence in train_sentences for word in sentence)
    ranked = sorted(counts, key=lambda word: (-counts[word], word))
    vocab = {"<UNK>": 0}
    for word in ranked[:4000]:
        vocab[word] = len(vocab)
    return vocab


def encode_sentences(sentences, vocab):
    return [[vocab.get(word, vocab["<UNK>"]) for word in sentence]
            for sentence in sentences]


def make_batches(sentences, batch_size=4):
    by_length = defaultdict(list)
    for index, sentence in enumerate(sentences):
        by_length[len(sentence)].append(index)

    batches = []
    for length in sorted(by_length):
        indices = by_length[length]
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = torch.tensor([sentences[i] for i in batch_indices], dtype=torch.long)
            batches.append((batch, batch_indices))
    return batches


class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.bilstm = nn.LSTM(
            input_size=256, hidden_size=256, num_layers=1,
            bidirectional=True, batch_first=True,
        )
        self.to_params = nn.Linear(512, 128)

    def forward(self, word_ids):
        word_vectors = self.embedding(word_ids)
        hidden_states, _ = self.bilstm(word_vectors)
        sentence_vector = hidden_states.max(dim=1).values
        mean, logvar = self.to_params(sentence_vector).chunk(2, dim=-1)
        return mean, logvar


class GrammarScorer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.num_nt = 30
        self.num_pt = 60
        self.num_symbols = self.num_nt + self.num_pt

        self.root_embedding = nn.Parameter(torch.randn(256) * 0.02)
        self.nt_embedding = nn.Embedding(self.num_nt, 256)
        self.pt_embedding = nn.Embedding(self.num_pt, 256)

        def mlp(output_size):
            return nn.Sequential(
                nn.Linear(256 + 64, 256),
                nn.ReLU(),
                nn.Linear(256, output_size),
            )

        self.root_mlp = mlp(self.num_nt)
        self.rule_mlp = mlp(self.num_symbols ** 2)
        self.word_mlp = mlp(vocab_size)

    def forward(self, z):
        batch_size = z.size(0)

        root = self.root_embedding.unsqueeze(0).expand(batch_size, -1)
        root_input = torch.cat([root, z], dim=-1)
        root_log_probs = F.log_softmax(self.root_mlp(root_input), dim=-1)

        nt = self.nt_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        nt_input = torch.cat(
            [nt, z.unsqueeze(1).expand(-1, self.num_nt, -1)], dim=-1
        )
        binary_log_probs = F.log_softmax(
            self.rule_mlp(nt_input), dim=-1
        ).view(batch_size, self.num_nt, self.num_symbols, self.num_symbols)

        pt = self.pt_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        pt_input = torch.cat(
            [pt, z.unsqueeze(1).expand(-1, self.num_pt, -1)], dim=-1
        )
        word_log_probs = F.log_softmax(self.word_mlp(pt_input), dim=-1)

        return root_log_probs, binary_log_probs, word_log_probs


def choose_device(choice):
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or auto.")
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def sample_z(mean, logvar):
    return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)


def sentence_terms(batch, word_log):
    indices = batch.unsqueeze(1).expand(-1, word_log.size(1), -1)
    return word_log.gather(2, indices).transpose(1, 2).contiguous()


def losses_for_batch(batch, encoder, grammar):
    mean, logvar = encoder(batch)
    root_log, binary_log, word_log = grammar(sample_z(mean, logvar))
    terms = sentence_terms(batch, word_log)
    nll = -SentCFG((terms, binary_log, root_log)).partition
    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1)
    return nll, kl


@torch.no_grad()
def validation_loss(batches, encoder, grammar, device):
    encoder.eval()
    grammar.eval()
    total = count = 0
    gpu_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=gpu_devices):
        torch.manual_seed(1234)
        for batch, _ in batches:
            batch = batch.to(device)
            nll, kl = losses_for_batch(batch, encoder, grammar)
            total += (nll + kl).sum().item()
            count += batch.size(0)
    return total / count


def read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Put the saved .pt file in "
            "checkpoints/ or pass its location with --checkpoint."
        )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("encoder", "grammar", "vocab"):
        if key not in saved:
            raise ValueError(f"Checkpoint is missing '{key}': {path}")
    return saved


def find_checkpoint(path=None):
    if path is not None:
        checkpoint = Path(path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    project_root = Path(__file__).resolve().parent.parent
    names = (
        "checkpoints/task2_best_epoch1_saved.pt",
        "checkpoints/task2_best.pt",
        "task2_best_epoch1_saved.pt",
        "checkpoints/task2_batch8/task2_best.pt",
    )
    candidates = [base / name for base in (Path.cwd(), project_root) for name in names]
    candidates.extend((
        Path("/content/drive/MyDrive/From_CKY_to_Neural_Grammar_Induction/")
        / "task2_best_epoch1_saved.pt",
        Path("/content/drive/MyDrive/From_CKY_to_Neural_Grammar_Induction/")
        / "task2_best.pt",
    ))

    for checkpoint in candidates:
        if checkpoint.is_file():
            return checkpoint
    raise FileNotFoundError(
        "No Task 2 checkpoint found. Put task2_best_epoch1_saved.pt in "
        "checkpoints/, supply --checkpoint PATH, or train with --train."
    )


def make_models(vocab_size, device):
    encoder = SentenceEncoder(vocab_size).to(device)
    grammar = GrammarScorer(vocab_size).to(device)
    return encoder, grammar


def train(args, device):
    groups = load_partition_a()
    train_sentences = [t.leaves() for t in groups["train"] if len(t.leaves()) >= 2]
    validation_sentences = [t.leaves() for t in groups["validation"]]
    assert len(train_sentences) == 1043
    vocab = build_vocab(train_sentences)
    train_ids = encode_sentences(train_sentences, vocab)
    validation_ids = encode_sentences(validation_sentences, vocab)
    train_batches = make_batches(train_ids, args.batch_size)
    validation_batches = make_batches(validation_ids, 4)

    torch.manual_seed(42)
    encoder, grammar = make_models(len(vocab), device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(grammar.parameters()),
        lr=1e-3, betas=(0.75, 0.999),
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    if args.resume:
        saved = read_checkpoint(args.resume)
        if saved["vocab"] != vocab:
            raise ValueError("Checkpoint vocabulary does not match Partition A.")
        encoder.load_state_dict(saved["encoder"])
        grammar.load_state_dict(saved["grammar"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"] + 1
        best_validation_loss = saved["best_validation_loss"]

    if start_epoch > args.epochs:
        raise ValueError(f"Checkpoint is already at epoch {start_epoch - 1}; increase --epochs.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | train: {len(train_ids)} | validation: {len(validation_ids)}")
    print(f"Batch size: {args.batch_size} | batches: {len(train_batches)}")

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        grammar.train()
        order = list(range(len(train_batches)))
        random.Random(42 + epoch).shuffle(order)
        total = count = 0

        for step, batch_number in enumerate(order, start=1):
            batch, _ = train_batches[batch_number]
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            nll, kl = losses_for_batch(batch, encoder, grammar)
            loss = (nll + kl).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {batch_number}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(grammar.parameters()), 3.0
            )
            optimizer.step()
            total += loss.item() * batch.size(0)
            count += batch.size(0)

            if step % 25 == 0:
                print(f"Epoch {epoch}: {step}/{len(train_batches)} batches", flush=True)

        val_loss = validation_loss(validation_batches, encoder, grammar, device)
        improved = val_loss < best_validation_loss
        if improved:
            best_validation_loss = val_loss

        saved = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "grammar": grammar.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vocab": vocab,
            "best_validation_loss": best_validation_loss,
        }
        torch.save(saved, output_dir / "task2_latest.pt")
        if improved:
            torch.save(saved, output_dir / "task2_best.pt")
        print(
            f"Epoch {epoch:02d} | train loss {total / count:.3f} "
            f"| validation loss {val_loss:.3f} | sentences {count}",
            flush=True,
        )

    best_path = output_dir / "task2_best.pt"
    if best_path.exists():
        print("Best checkpoint:", best_path)
    else:
        print("No better checkpoint was saved in this resumed run.")
        print("Previous checkpoint:", args.resume)


def model_potentials(sentence_ids, encoder, grammar, device):
    batch = torch.tensor([sentence_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        mean, _ = encoder(batch)
        root_log, binary_log, word_log = grammar(mean)
        terms = sentence_terms(batch, word_log)
    return terms.detach(), binary_log.detach(), root_log.detach()


def decode_spans(sentence_ids, encoder, grammar, device):
    """Viterbi tree, with the sentence root omitted from the returned spans."""
    if len(sentence_ids) < 2:
        raise ValueError("A sentence needs at least two words for this grammar.")
    terms, binary_log, root_log = model_potentials(sentence_ids, encoder, grammar, device)
    with torch.enable_grad():
        span_mask = SentCFG((terms, binary_log, root_log)).argmax[3][0].detach()

    n = len(sentence_ids)
    return Counter(
        (start, start + width_index + 2)
        for width_index, start, symbol in (span_mask > 0.5).nonzero(as_tuple=False).tolist()
        if not (start == 0 and start + width_index + 2 == n)
    )


def decode_spans_mbr(sentence_ids, encoder, grammar, device):
    """Build the binary tree with maximum sum of unlabeled span marginals."""
    n = len(sentence_ids)
    if n < 2:
        raise ValueError("A sentence needs at least two words for this grammar.")

    terms, binary_log, root_log = model_potentials(sentence_ids, encoder, grammar, device)
    with torch.enable_grad():
        marginals = SentCFG((terms, binary_log, root_log)).marginals[3][0]
    span_scores = marginals.sum(dim=-1).detach().cpu()

    best = [[0.0] * (n + 1) for _ in range(n + 1)]
    split = [[None] * (n + 1) for _ in range(n + 1)]
    for width in range(2, n + 1):
        for start in range(n - width + 1):
            end = start + width
            middle = max(
                range(start + 1, end),
                key=lambda k: best[start][k] + best[k][end],
            )
            reward = 0.0 if width == n else span_scores[width - 2, start].item()
            best[start][end] = best[start][middle] + best[middle][end] + reward
            split[start][end] = middle

    spans = Counter()

    def collect(start, end):
        if end - start < 2:
            return
        if not (start == 0 and end == n):
            spans[(start, end)] += 1
        middle = split[start][end]
        collect(start, middle)
        collect(middle, end)

    collect(0, n)
    return spans


def get_brackets(tree):
    brackets = Counter()

    def visit(node, start, is_root=False):
        if isinstance(node, str):
            return start + 1
        end = start
        for child in node:
            end = visit(child, end)
        if not is_root and any(isinstance(child, Tree) for child in node):
            brackets[(start, end)] += 1
        return end

    visit(tree, 0, is_root=True)
    return brackets


def evaluation_spans(spans, gold_tree):
    keep = [tag not in PUNCT_TAGS for word, tag in gold_tree.pos()]
    prefix = [0]
    for keep_word in keep:
        prefix.append(prefix[-1] + int(keep_word))

    sentence_length = prefix[-1]
    result = set()
    for start, end in spans:
        new_start, new_end = prefix[start], prefix[end]
        if 2 <= new_end - new_start < sentence_length:
            result.add((new_start, new_end))
    return result


def spans_to_tree(words, spans):
    """Display an induced binary parse; X is an unlabeled constituent."""
    n = len(words)
    selected = set(spans) | {(0, n)}

    def build(start, end):
        if end - start == 1:
            return words[start]
        for middle in range(start + 1, end):
            if ((middle == start + 1 or (start, middle) in selected)
                    and (middle == end - 1 or (middle, end) in selected)):
                return Tree("X", [build(start, middle), build(middle, end)])
        raise ValueError("Decoded spans do not form a binary tree.")

    return build(0, n)


def evaluate(ids, gold_trees, decoder, encoder, grammar, device, show_tree=0, limit=None):
    sentence_scores = []
    correct = predicted_total = gold_total = 0
    one_sided = long_enough = skipped = 0
    pairs = list(zip(ids, gold_trees))
    if limit is not None:
        pairs = pairs[:limit]

    for index, (sentence_ids, gold_tree) in enumerate(pairs):
        assert len(sentence_ids) == len(gold_tree.leaves())
        if sum(tag not in PUNCT_TAGS for _, tag in gold_tree.pos()) < 2:
            skipped += 1
            continue

        predicted_raw = decoder(sentence_ids, encoder, grammar, device)
        predicted = evaluation_spans(predicted_raw, gold_tree)
        gold = evaluation_spans(get_brackets(gold_tree), gold_tree)
        matches = len(predicted & gold)
        correct += matches
        predicted_total += len(predicted)
        gold_total += len(gold)
        sentence_scores.append(
            2 * matches / (len(predicted) + len(gold))
            if predicted or gold else 1.0
        )

        if len(sentence_ids) >= 8:
            endings = Counter()
            for (_, end), count in predicted_raw.items():
                endings[end] += count
            if endings and max(endings.values()) / sum(endings.values()) >= 0.8:
                one_sided += 1
            long_enough += 1

        if show_tree > 0:
            words = gold_tree.leaves()
            print(f"\nSentence {index}: {' '.join(words)}")
            print("Predicted unlabeled tree:")
            print(spans_to_tree(words, predicted_raw).pformat(margin=90))
            print("Gold tree:")
            print(gold_tree.pformat(margin=90))
            show_tree -= 1

        if (index + 1) % 20 == 0:
            print(f"Processed {index + 1}/{len(pairs)}", flush=True)

    sentence_f1 = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
    count = predicted_total + gold_total
    corpus_f1 = 2 * correct / count if count else 0.0
    print(f"Sentences scored: {len(sentence_scores)}/{len(pairs)} | skipped: {skipped}")
    print(f"Unlabeled sentence F1: {sentence_f1:.3f}")
    print(f"Unlabeled corpus F1: {corpus_f1:.3f}")
    print(f"One-sided trees: {one_sided}/{long_enough} (8+ words)")
    if limit is not None:
        print("This is a subset check; run without --limit for the full split.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Train a new model (slow on CPU).")
    parser.add_argument("--checkpoint", type=Path,
                        help="Saved .pt file; if omitted, search checkpoints/ and mounted Drive.")
    parser.add_argument("--resume", type=Path, help="Continue training from a saved checkpoint.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--decoder", choices=("viterbi", "mbr"), default="mbr")
    parser.add_argument("--limit", type=int, help="Evaluate just the first N sentences for a quick check.")
    parser.add_argument("--show-tree", type=int, default=0, help="Print N predicted and gold trees.")
    parser.add_argument("--text", help="Parse one already tokenized, space-separated sentence.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (default: 8).")
    parser.add_argument("--epochs", type=int, default=15, help="Total training epochs (default: 15).")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/task2_batch8"))
    args = parser.parse_args()

    if args.train and args.checkpoint:
        parser.error("Use --resume to continue training, or --checkpoint to evaluate.")
    if args.train and args.text:
        parser.error("Use --text for evaluation, without --train.")
    if args.resume and not args.train:
        parser.error("--resume requires --train.")
    if args.batch_size < 1 or args.epochs < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size, --epochs, and --limit must be positive.")
    if args.show_tree < 0:
        parser.error("--show-tree must be zero or positive.")

    device = choose_device(args.device)
    print("Device:", device)
    if args.train:
        if device.type == "cpu":
            print("Training on CPU may take a long time. A CUDA machine can use --device auto.")
        train(args, device)
        return

    checkpoint_path = find_checkpoint(args.checkpoint)
    saved = read_checkpoint(checkpoint_path)
    print("Loaded checkpoint:", checkpoint_path)
    vocab = saved["vocab"]
    torch.manual_seed(42)
    encoder, grammar = make_models(len(vocab), device)
    encoder.load_state_dict(saved["encoder"])
    grammar.load_state_dict(saved["grammar"])
    encoder.eval()
    grammar.eval()
    print("Loaded checkpoint epoch:", saved.get("epoch", "unknown"))

    decoder = decode_spans_mbr if args.decoder == "mbr" else decode_spans
    if args.text:
        words = args.text.split()
        if len(words) < 2:
            parser.error("--text needs at least two space-separated tokens.")
        sentence_ids = encode_sentences([words], vocab)[0]
        predicted = decoder(sentence_ids, encoder, grammar, device)
        print("Unlabeled parse (X marks an induced constituent):")
        print(spans_to_tree(words, predicted).pformat(margin=90))
        return

    groups = load_partition_a()
    train_sentences = [t.leaves() for t in groups["train"] if len(t.leaves()) >= 2]
    if build_vocab(train_sentences) != vocab:
        raise ValueError("Saved vocabulary does not match these Partition A training sentences.")
    gold_trees = groups[args.split]
    ids = encode_sentences([t.leaves() for t in gold_trees], vocab)
    print(f"Evaluating {args.split} with {args.decoder.upper()} ({len(gold_trees)} sentences).")
    evaluate(ids, gold_trees, decoder, encoder, grammar, device, args.show_tree, args.limit)


if __name__ == "__main__":
    main()
