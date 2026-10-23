from collections import Counter, defaultdict
from math import log

from nltk.grammar import Nonterminal
from nltk.tree import Tree


def replace_rare_words(trees, cutoff=1):
    word_counts = Counter(word for tree in trees for word in tree.leaves())
    changed_trees = []

    for tree in trees:
        changed = tree.copy(deep=True)
        for position in changed.treepositions("leaves"):
            if word_counts[changed[position]] <= cutoff:
                changed[position] = "<UNK>"
        changed_trees.append(changed)

    return changed_trees


def to_cnf(tree, markov_order=0):
    converted = tree.copy(deep=True)
    converted.collapse_unary(collapsePOS=True, collapseRoot=True)
    converted.chomsky_normal_form(
        factor="right",
        horzMarkov=markov_order,
    )
    return converted


def is_cnf(tree):
    for node in tree.subtrees():
        binary = (
            len(node) == 2
            and all(isinstance(child, Tree) for child in node)
        )
        lexical = len(node) == 1 and isinstance(node[0], str)

        if not (binary or lexical):
            return False

    return True


def count_rules(cnf_trees):
    counts = Counter()

    for tree in cnf_trees:
        counts.update(tree.productions())

    return counts


def estimate_rule_probabilities(counts):
    lhs_totals = Counter()
    rare_by_lhs = Counter()

    for rule, count in counts.items():
        lhs = rule.lhs()
        lhs_totals[lhs] += count

        if count < 5:
            rare_by_lhs[lhs] += 1

    probabilities = {}

    for rule, count in counts.items():
        lhs = rule.lhs()
        adjusted_count = count + 1 if count < 5 else count
        adjusted_total = lhs_totals[lhs] + rare_by_lhs[lhs]
        probabilities[rule] = adjusted_count / adjusted_total

    return probabilities

def index_rules(probabilities):
    lexical_rules = defaultdict(list)
    binary_rules = defaultdict(list)

    for rule, probability in probabilities.items():
        lhs = rule.lhs()
        rhs = rule.rhs()
        score = log(probability)

        if len(rhs) == 1 and isinstance(rhs[0], str):
            lexical_rules[rhs[0]].append((lhs, score))
        elif len(rhs) == 2:
            binary_rules[(rhs[0], rhs[1])].append((lhs, score))
        else:
            raise ValueError(f"Rule is not in CNF: {rule}")

    return lexical_rules, binary_rules


def build_tree(chart, i, j, label):
    _, choice = chart[i][j][label]

    if choice[0] == "word":
        return Tree(label.symbol(), [choice[1]])

    _, split, left_label, right_label = choice

    return Tree(label.symbol(), [
        build_tree(chart, i, split, left_label),
        build_tree(chart, split, j, right_label),
    ])


def parse_sentence(words, lexical_rules, binary_rules, root_counts):
    n = len(words)
    if n == 0 or not root_counts:
        return None

    chart = [[{} for _ in range(n + 1)] for _ in range(n)]

    for i, word in enumerate(words):
        lookup_word = word if word in lexical_rules else "<UNK>"
        rules = lexical_rules.get(lookup_word, [])
        if not rules:
            return None

        for lhs, score in rules:
            chart[i][i + 1][lhs] = (score, ("word", word))

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length

            for split in range(i + 1, j):
                for left, (left_score, _) in chart[i][split].items():
                    for right, (right_score, _) in chart[split][j].items():
                        for lhs, rule_score in binary_rules.get(
                            (left, right), []
                        ):
                            score = left_score + right_score + rule_score
                            previous = chart[i][j].get(lhs)

                            if previous is None or score > previous[0]:
                                chart[i][j][lhs] = (
                                    score,
                                    ("split", split, left, right),
                                )

    best_root = None
    best_score = float("-inf")
    total_roots = sum(root_counts.values())

    for label, count in root_counts.items():
        root = Nonterminal(label)

        if root in chart[0][n]:
            score = chart[0][n][root][0] + log(count / total_roots)

            if score > best_score:
                best_score = score
                best_root = root

    if best_root is None:
        return None

    tree = build_tree(chart, 0, n, best_root)
    tree.un_chomsky_normal_form()
    return tree


def get_brackets(tree, labeled=True):
    brackets = Counter()

    def visit(node, start, is_root=False):
        if isinstance(node, str):
            return start + 1

        end = start
        for child in node:
            end = visit(child, end)

        if not is_root and any(isinstance(child, Tree) for child in node):
            key = (start, end, node.label()) if labeled else (start, end)
            brackets[key] += 1

        return end

    visit(tree, 0, is_root=True)
    return brackets


def evaluate_parser(test_trees, lexical_rules, binary_rules, root_counts,
                    training_vocabulary):
    totals = {
        "labeled": Counter(),
        "unlabeled": Counter(),
    }
    parsed_count = 0
    unseen_word_sentences = 0
    sentences_using_unk = 0
    test_predictions = []

    for gold_tree in test_trees:
        words = gold_tree.leaves()

        if any(word not in training_vocabulary for word in words):
            unseen_word_sentences += 1
        if any(word not in lexical_rules for word in words):
            sentences_using_unk += 1

        predicted_tree = parse_sentence(
            words, lexical_rules, binary_rules, root_counts
        )
        test_predictions.append(predicted_tree)

        if predicted_tree is not None:
            parsed_count += 1
            assert predicted_tree.leaves() == words

        for name, labeled in (("labeled", True), ("unlabeled", False)):
            gold = get_brackets(gold_tree, labeled)
            predicted = (
                get_brackets(predicted_tree, labeled)
                if predicted_tree is not None
                else Counter()
            )

            totals[name]["correct"] += sum((predicted & gold).values())
            totals[name]["predicted"] += sum(predicted.values())
            totals[name]["gold"] += sum(gold.values())

    results = {
        "test_sentences": len(test_trees),
        "sentences_with_unseen_words": unseen_word_sentences,
        "sentences_using_unk": sentences_using_unk,
        "complete_parses": parsed_count,
    }

    for name, counts in totals.items():
        precision = (
            counts["correct"] / counts["predicted"]
            if counts["predicted"] else 0
        )
        recall = counts["correct"] / counts["gold"] if counts["gold"] else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return results, test_predictions


def print_results(split_name, results):
    print(f"{split_name} sentences: {results['test_sentences']}")
    print("Sentences with words absent from training: "
          f"{results['sentences_with_unseen_words']}")
    print(f"Sentences using <UNK>: {results['sentences_using_unk']}")
    print(f"Complete parses: {results['complete_parses']}/{results['test_sentences']}")

    for name in ("labeled", "unlabeled"):
        scores = results[name]
        print(f"{name.capitalize()}: P={scores['precision']:.3f}, "
              f"R={scores['recall']:.3f}, F1={scores['f1']:.3f}")


if __name__ == "__main__":
    from data import train_trees, validation_trees, test_trees

    training_vocabulary = {
        word for tree in train_trees for word in tree.leaves()
    }
    models = {}
    validation_results = {}

    for cutoff in (1, 2):
        train_for_pcfg = replace_rare_words(train_trees, cutoff=cutoff)
        train_cnf_0 = [to_cnf(tree, markov_order=0) for tree in train_for_pcfg]

        rule_counts = count_rules(train_cnf_0)
        rule_probabilities = estimate_rule_probabilities(rule_counts)
        lexical_rules, binary_rules = index_rules(rule_probabilities)
        root_counts = Counter(tree.label() for tree in train_cnf_0)

        models[cutoff] = (lexical_rules, binary_rules, root_counts)
        results, _ = evaluate_parser(
            validation_trees, lexical_rules, binary_rules, root_counts,
            training_vocabulary,
        )
        validation_results[cutoff] = results

        print(f"\nMarkov order 0, cutoff {cutoff}")
        print_results("Validation", results)

    best_cutoff = max(
        validation_results,
        key=lambda cutoff: validation_results[cutoff]["labeled"]["f1"],
    )
    print(f"\nChosen cutoff from validation F1: {best_cutoff}")

    lexical_rules, binary_rules, root_counts = models[best_cutoff]
    test_results, test_predictions = evaluate_parser(
        test_trees, lexical_rules, binary_rules, root_counts,
        training_vocabulary,
    )
    print_results("Test", test_results)
