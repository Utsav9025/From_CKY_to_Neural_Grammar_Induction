from collections import Counter

from nltk.tree import Tree


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