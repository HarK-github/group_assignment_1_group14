"""
Question 4, Part 2: PCFG Induction and CKY Parser

This module provides:
1. PCFG induction from Penn Treebank trees.
2. A Viterbi CKY parser for finding the most probable parse.
3. POS-tag reconciliation between an HMM tagger and the PCFG parser.
"""

from typing import List, Tuple, Any, Dict, Optional
from collections import defaultdict

import nltk
from nltk import Tree
from nltk.corpus import treebank as nltk_treebank_corpus
from nltk.grammar import PCFG, Nonterminal
from nltk.tree import chomsky_normal_form


# ---------------------------------------------------------------------------
# 1. PCFG TRAINING
# ---------------------------------------------------------------------------

def train_pcfg(treebank: Any = None) -> PCFG:
    """
    Trains a PCFG using Penn Treebank data.

    Args:
        treebank:
            A list of nltk.Tree objects.
            If None, the NLTK Penn Treebank corpus is used.

    Returns:
        A trained NLTK PCFG.
    """

    if treebank is None:
        treebank = nltk_treebank_corpus.parsed_sents()

    if not treebank:
        raise ValueError("Treebank is empty; cannot train PCFG.")

    productions = []

    for tree in treebank:
        # Work on a copy so the original treebank is not modified.
        tree = tree.copy(deep=True)

        # Convert trees to Chomsky Normal Form so they can be
        # handled by the CKY algorithm.
        chomsky_normal_form(
            tree,
            factor="right",
            horzMarkov=2,
            vertMarkov=0
        )

        productions.extend(tree.productions())

    # The first production's LHS is the start symbol.
    start_symbol = productions[0].lhs()

    # Estimate rule probabilities using maximum likelihood.
    pcfg = nltk.induce_pcfg(
        start_symbol,
        productions
    )

    return pcfg


# ---------------------------------------------------------------------------
# 2. VITERBI / CKY PARSER
# ---------------------------------------------------------------------------

def cky_parse(
    sentence: List[str],
    pcfg: PCFG
) -> Optional[Tree]:
    """
    Parses a sentence using the Viterbi CKY algorithm.

    Args:
        sentence:
            A list of tokens.

        pcfg:
            A trained PCFG.

    Returns:
        The highest-probability parse tree, or None if the sentence
        cannot be parsed by the grammar.
    """

    n = len(sentence)

    if n == 0:
        return None

    # -----------------------------------------------------------------------
    # Separate grammar rules into binary, lexical and unary rules.
    #
    # Binary:
    #     A -> B C
    #
    # Lexical:
    #     A -> "word"
    #
    # Unary:
    #     A -> B
    # -----------------------------------------------------------------------

    binary_rules = defaultdict(list)
    lexical_rules = defaultdict(list)
    unary_rules = defaultdict(list)

    for production in pcfg.productions():

        lhs = production.lhs()
        rhs = production.rhs()
        probability = production.prob()

        # A -> B C
        if (
            len(rhs) == 2
            and isinstance(rhs[0], Nonterminal)
            and isinstance(rhs[1], Nonterminal)
        ):
            binary_rules[
                (rhs[0], rhs[1])
            ].append(
                (lhs, probability)
            )

        # A -> "word"
        elif (
            len(rhs) == 1
            and isinstance(rhs[0], str)
        ):
            lexical_rules[
                rhs[0]
            ].append(
                (lhs, probability)
            )

        # A -> B
        elif (
            len(rhs) == 1
            and isinstance(rhs[0], Nonterminal)
        ):
            unary_rules[
                rhs[0]
            ].append(
                (lhs, probability)
            )

    # -----------------------------------------------------------------------
    # CKY chart.
    #
    # chart[i][j][X] =
    #     (best_probability, backpointer)
    #
    # Represents the best derivation of nonterminal X over
    # the span sentence[i:j].
    # -----------------------------------------------------------------------

    chart = [
        [dict() for _ in range(n + 1)]
        for _ in range(n)
    ]

    # -----------------------------------------------------------------------
    # Unary closure
    # -----------------------------------------------------------------------

    def apply_unary_closure(i: int, j: int) -> None:
        """
        Repeatedly applies unary rules until no better derivation
        can be found for the current span.
        """

        changed = True

        while changed:
            changed = False

            current_symbols = list(
                chart[i][j].keys()
            )

            for child_symbol in current_symbols:

                child_prob, _ = chart[i][j][child_symbol]

                for parent_symbol, rule_prob in unary_rules.get(
                    child_symbol,
                    []
                ):

                    new_prob = (
                        child_prob
                        * rule_prob
                    )

                    if (
                        parent_symbol not in chart[i][j]
                        or new_prob
                        > chart[i][j][parent_symbol][0]
                    ):
                        chart[i][j][parent_symbol] = (
                            new_prob,
                            (
                                "unary",
                                child_symbol
                            )
                        )

                        changed = True

    # -----------------------------------------------------------------------
    # Initialization with lexical rules
    # -----------------------------------------------------------------------

    for i, word in enumerate(sentence):

        # Exact word match first.
        if word in lexical_rules:
            lexical_entries = lexical_rules[word]

        # Try lowercase as a convenience for grammars whose lexical
        # entries are lowercase.
        elif word.lower() in lexical_rules:
            lexical_entries = lexical_rules[word.lower()]

        else:
            # No lexical rule can generate this word.
            continue

        for lhs, probability in lexical_entries:

            chart[i][i + 1][lhs] = (
                probability,
                (
                    "terminal",
                    word
                )
            )

        apply_unary_closure(i, i + 1)

    # -----------------------------------------------------------------------
    # Main CKY dynamic-programming algorithm
    # -----------------------------------------------------------------------

    for span_length in range(2, n + 1):

        for i in range(
            0,
            n - span_length + 1
        ):

            j = i + span_length

            # Try every possible split point.
            for k in range(i + 1, j):

                left_symbols = list(
                    chart[i][k].keys()
                )

                right_symbols = list(
                    chart[k][j].keys()
                )

                for left_symbol in left_symbols:

                    left_prob, _ = chart[i][k][left_symbol]

                    for right_symbol in right_symbols:

                        right_prob, _ = chart[k][j][right_symbol]

                        rules = binary_rules.get(
                            (
                                left_symbol,
                                right_symbol
                            ),
                            []
                        )

                        for parent_symbol, rule_prob in rules:

                            new_prob = (
                                left_prob
                                * right_prob
                                * rule_prob
                            )

                            # Viterbi: keep only the best derivation.
                            if (
                                parent_symbol not in chart[i][j]
                                or new_prob
                                > chart[i][j][parent_symbol][0]
                            ):
                                chart[i][j][parent_symbol] = (
                                    new_prob,
                                    (
                                        "binary",
                                        left_symbol,
                                        right_symbol,
                                        k
                                    )
                                )

            apply_unary_closure(i, j)

    # -----------------------------------------------------------------------
    # Check whether the start symbol spans the entire sentence.
    # -----------------------------------------------------------------------

    start_symbol = pcfg.start()

    if start_symbol not in chart[0][n]:
        # Sentence cannot be generated by this grammar.
        return None

    # -----------------------------------------------------------------------
    # Reconstruct the best parse tree.
    # -----------------------------------------------------------------------

    def build_tree(
        symbol: Nonterminal,
        i: int,
        j: int
    ) -> Tree:

        _, backpointer = chart[i][j][symbol]

        pointer_type = backpointer[0]

        # A -> word
        if pointer_type == "terminal":

            word = backpointer[1]

            return Tree(
                symbol.symbol(),
                [word]
            )

        # A -> B
        elif pointer_type == "unary":

            child_symbol = backpointer[1]

            child_tree = build_tree(
                child_symbol,
                i,
                j
            )

            return Tree(
                symbol.symbol(),
                [child_tree]
            )

        # A -> B C
        elif pointer_type == "binary":

            (
                _,
                left_symbol,
                right_symbol,
                split
            ) = backpointer

            left_tree = build_tree(
                left_symbol,
                i,
                split
            )

            right_tree = build_tree(
                right_symbol,
                split,
                j
            )

            return Tree(
                symbol.symbol(),
                [
                    left_tree,
                    right_tree
                ]
            )

        raise ValueError(
            f"Unknown backpointer type: {pointer_type}"
        )

    return build_tree(
        start_symbol,
        0,
        n
    )


# ---------------------------------------------------------------------------
# 3. PARSE TREE + PROBABILITY
# ---------------------------------------------------------------------------

def cky_parse_with_probability(
    sentence: List[str],
    pcfg: PCFG
) -> Tuple[Optional[Tree], float]:
    """
    Parses a sentence and returns its most probable parse
    together with the probability of that parse.

    Args:
        sentence:
            A list of tokens.

        pcfg:
            A trained PCFG.

    Returns:
        (parse_tree, probability)

        If the sentence is unparseable:
            (None, 0.0)
    """

    tree = cky_parse(
        sentence,
        pcfg
    )

    if tree is None:
        return None, 0.0

    # Build a direct rule -> probability lookup.
    rule_probabilities = {
        (
            production.lhs(),
            tuple(production.rhs())
        ): production.prob()
        for production in pcfg.productions()
    }

    probability = 1.0

    for production in tree.productions():

        key = (
            production.lhs(),
            tuple(production.rhs())
        )

        if key not in rule_probabilities:
            # Should not normally happen for a tree produced
            # by this grammar.
            return tree, 0.0

        probability *= rule_probabilities[key]

    return tree, probability


# ---------------------------------------------------------------------------
# 4. POS TAG RECONCILIATION
# ---------------------------------------------------------------------------

# Common Brown -> Penn Treebank conversions.
#
# Tags not present in this table are left unchanged and normalized
# where appropriate.
BROWN_TO_PTB = {

    # Adjectives
    "jj": "JJ",
    "jjt": "JJR",
    "jjs": "JJS",

    # Nouns
    "nn": "NN",
    "nns": "NNS",
    "np": "NNP",
    "nps": "NNPS",

    # Verbs
    "vb": "VB",
    "vbd": "VBD",
    "vbg": "VBG",
    "vbn": "VBN",
    "vbz": "VBZ",

    # Modal
    "md": "MD",

    # Adverbs
    "rb": "RB",
    "rbr": "RBR",
    "rbs": "RBS",

    # Pronouns
    "pp$": "PRP$",
    "ppo": "PRP",
    "pn": "PRP",

    # Determiners
    "at": "DT",
    "dt": "DT",

    # Conjunction
    "cc": "CC",

    # Preposition
    "in": "IN",

    # Particle
    "rp": "RP",

    # To
    "to": "TO",

    # Possessive
    "pos": "POS",

    # Interjection
    "uh": "UH",

    # Cardinal number
    "cd": "CD",

    # Foreign word
    "fw": "FW",

    # Existential
    "ex": "EX",

    # Wh words
    "wdt": "WDT",
    "wp": "WP",
    "wpo": "WP",
    "wrb": "WRB",
    "wq": "WP",

    # Punctuation
    ".": ".",
    ",": ",",
    ":": ":",
    ";": ":",
    "(": "-LRB-",
    ")": "-RRB-",
}


def reconcile_tags(
    pos_tags: List[str],
    pcfg_tags: List[str]
) -> List[str]:
    """
    Reconciles POS tags from the HMM tagger and the PCFG parser.

    Strategy:
        Brown/HMM tags are converted into the Penn Treebank tagset.
        When the PCFG parser has supplied a tag, the PCFG tag is preferred
        because the grammar itself was trained on Penn Treebank labels.

    Args:
        pos_tags:
            POS tags produced by the HMM tagger.

        pcfg_tags:
            POS tags produced by the CKY parser.

    Returns:
        A reconciled list of POS tags.

    Raises:
        ValueError:
            If the two tag sequences have different lengths.
    """

    if len(pos_tags) != len(pcfg_tags):
        raise ValueError(
            "POS tag sequences must have the same length."
        )

    reconciled = []

    for hmm_tag, parser_tag in zip(
        pos_tags,
        pcfg_tags
    ):

        # Convert Brown-style HMM tag to PTB.
        normalized_hmm_tag = hmm_tag.lower()

        converted_hmm_tag = BROWN_TO_PTB.get(
            normalized_hmm_tag,
            hmm_tag.upper()
        )

        # Prefer the PCFG tag when available because it belongs
        # to the same tagset used to train the PCFG.
        if parser_tag:
            reconciled.append(parser_tag)
        else:
            reconciled.append(converted_hmm_tag)

    return reconciled
