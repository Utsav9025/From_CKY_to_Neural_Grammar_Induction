from collections import Counter, defaultdict

import torch
from torch import nn


def encode_sentences(sentences, vocab):
    unk_id = vocab["<UNK>"]
    return [
        [vocab.get(word, unk_id) for word in sentence]
        for sentence in sentences
    ]


def make_batches(sentences, batch_size=4):
    by_length = defaultdict(list)
    for index, sentence in enumerate(sentences):
        by_length[len(sentence)].append(index)

    batches = []
    for length in sorted(by_length):
        indices = by_length[length]
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = torch.tensor(
                [sentences[i] for i in batch_indices], dtype=torch.long
            )
            batches.append((batch, batch_indices))

    return batches


class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.to_params = nn.Linear(512, 128)

    def forward(self, word_ids):
        word_vectors = self.embedding(word_ids)
        hidden_states, _ = self.bilstm(word_vectors)
        sentence_vector = hidden_states.max(dim=1).values
        mean, logvar = self.to_params(sentence_vector).chunk(2, dim=-1)
        return mean, logvar


def main():
    # data.py creates the same Partition A splits and cleaned trees as the notebook.
    from data import train_trees, validation_trees, test_trees

    train_sentences = [tree.leaves() for tree in train_trees]
    validation_sentences = [tree.leaves() for tree in validation_trees]
    test_sentences = [tree.leaves() for tree in test_trees]

    print("Train:", len(train_sentences))
    print("Validation:", len(validation_sentences))
    print("Test:", len(test_sentences))
    print("Example:", train_sentences[0])

    model_train_sentences = [
        sentence for sentence in train_sentences if len(sentence) >= 2
    ]
    print("Original training sentences:", len(train_sentences))
    print("Usable for Compound PCFG:", len(model_train_sentences))

    word_counts = Counter(
        word for sentence in model_train_sentences for word in sentence
    )
    total_tokens = sum(word_counts.values())
    print("Training tokens:", total_tokens)
    print("Distinct training words:", len(word_counts))
    print("Words appearing once:", sum(n == 1 for n in word_counts.values()))
    print("Longest training sentence:", max(map(len, train_sentences)))
    print("10 most common:", word_counts.most_common(10))

    ranked_words = sorted(word_counts, key=lambda word: (-word_counts[word], word))
    vocab = {"<UNK>": 0}
    for word in ranked_words[:4000]:
        vocab[word] = len(vocab)

    unk_count = sum(
        count for word, count in word_counts.items() if word not in vocab
    )
    print("Words kept:", len(vocab) - 1)
    print("Training <UNK> occurrences:", unk_count)
    print("Training token coverage:", round(1 - unk_count / total_tokens, 3))

    train_ids = encode_sentences(model_train_sentences, vocab)
    validation_ids = encode_sentences(validation_sentences, vocab)
    test_ids = encode_sentences(test_sentences, vocab)

    print("First 10 words:", model_train_sentences[0][:10])
    print("Their IDs:", train_ids[0][:10])
    print("Same sentence length:", len(model_train_sentences[0]) == len(train_ids[0]))
    print(
        "Training <UNK> tokens:",
        sum(word_id == 0 for sentence in train_ids for word_id in sentence),
    )

    train_batches = make_batches(train_ids)
    validation_batches = make_batches(validation_ids)
    test_batches = make_batches(test_ids)
    print(
        "Training sentences in batches:",
        sum(batch.size(0) for batch, _ in train_batches),
    )
    print("Number of training batches:", len(train_batches))
    print("First batch shape:", tuple(train_batches[0][0].shape))

    for name, sentences in [
        ("train", train_sentences),
        ("validation", validation_sentences),
        ("test", test_sentences),
    ]:
        one_word = [
            (i, sentence) for i, sentence in enumerate(sentences)
            if len(sentence) == 1
        ]
        print(name, "one-word sentences:", len(one_word), one_word[:5])

    assert len(train_ids) == 1043, "Check the Partition A split and sentence filter"
    assert min(map(len, train_ids)) >= 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    encoder = SentenceEncoder(len(vocab)).to(device)

    sample_batch = next(batch for batch, _ in train_batches if batch.size(0) == 4)
    with torch.no_grad():
        mean, logvar = encoder(sample_batch.to(device))

    print("Device:", device)
    print("Input shape:", tuple(sample_batch.shape))
    print("Mean shape:", tuple(mean.shape))
    print("Log variance shape:", tuple(logvar.shape))


if __name__ == "__main__":
    main()