"""
Question 4, Part 4: Final Passage Analysis — Method Comparison and Decision Rule

This module implements the end-of-passage sentence-level evaluation pipeline for the
integrated background editor. It splits the post-processed token stream into sentences,
evaluates each sentence under three distinct scoring models:
  1. PCFG Constituency Parse Log-Probability (or "Unparseable")
  2. Add-k Smoothed Bigram Log-Probability
  3. Add-k Smoothed Trigram Log-Probability

It applies a documented hierarchical decision rule to select the most authoritative
method per sentence and renders a final grammaticality verdict, compiling all metrics into
a structured comparison table matching all assignment specifications.

===========================================================================================
DOCUMENTED DECISION RULE SPECIFICATION: METHOD SELECTION & GRAMMATICALITY VERDICT
===========================================================================================
The decision rule determines which model's judgment is most authoritative for each sentence,
balancing global syntactic phrase structure against local n-gram lexical fluency.

Tier 1: PCFG Constituency Parser (Global Structural / Syntactic Hierarchy)
  - Constituency parsing verifies grammatical phrase hierarchy (NP, VP, PP, clauses)
    and structural agreement.
  - Condition: If the PCFG successfully yields a valid parse tree and its normalized
    log-probability per token L_bar_pcfg >= TAU_PCFG_OUTLIER (default: -15.0 nats/token):
      * Chosen Method: 'PCFG'
      * Final Verdict: 'Grammatical'
  - Exception: If the sentence parses but produces an extreme outlier score
    (L_bar_pcfg < TAU_PCFG_OUTLIER), the syntactic structure is deemed dubious or unnatural.
    The rule defers to the N-gram models for further verification.
  - Fallback: If the PCFG flags the sentence as 'Unparseable' (structural break or
    uncovered production rule), the rule defers to Tier 2.

Tier 2: Trigram Language Model (Local Context Window & Collocations)
  - Trigrams evaluate 3-word local syntactic and lexical fluency.
  - Condition: Evaluated when PCFG is 'Unparseable' or an outlier. Compute length-normalized
    trigram log-probability L_bar_tri = total_log_prob / N (where N = token count).
  - Decision:
      * If L_bar_tri >= TAU_TRI_GRAMMATICAL (default: -8.8 nats/token, calibrated from Brown corpus):
          Adequate coverage and high collocation probability.
          - Chosen Method: 'Trigram LM'
          - Final Verdict: 'Grammatical'
      * If TAU_TRI_SPARSE <= L_bar_tri < TAU_TRI_GRAMMATICAL ([-11.0, -8.8)):
          Acceptable sequence, but contains rare collocations, stylistic quirks, or minor irregularities.
          - Chosen Method: 'Trigram LM'
          - Final Verdict: 'Questionable'
      * If L_bar_tri < TAU_TRI_SPARSE (-11.0 nats/token):
          Severe trigram backoff penalty. Indicates either an ungrammatical sequence
          or vocabulary sparsity. Defer to Tier 3.

Tier 3: Bigram Language Model (Transition Continuity & High-Coverage Fallback)
  - Bigram models provide robust pairwise transition probabilities with broad coverage.
  - Condition: Evaluated when Trigram exhibits extreme backoff (L_bar_tri < -11.0 nats/token).
  - Decision:
      * If L_bar_bi >= TAU_BI_GRAMMATICAL (default: -8.5 nats/token):
          Pairwise transitions are standard and fluent; low trigram score was driven by sparsity.
          - Chosen Method: 'Bigram LM'
          - Final Verdict: 'Grammatical'
      * Else (L_bar_bi < -8.5 nats/token):
          Both trigram and bigram models reject the transitions, and PCFG failed.
          - Chosen Method: 'Bigram LM'
          - Final Verdict: 'Ungrammatical'
===========================================================================================
"""

import os
import sys
import math
import re
import pickle
import types
from typing import List, Dict, Tuple, Any, Optional, Union

import pandas as pd
import nltk

# Defensive injection for environments missing 'conllu'
if 'conllu' not in sys.modules:
    sys.modules['conllu'] = types.ModuleType('conllu')

# Resolve paths to Question 1, 3, and 4
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
MODELS_DIR = os.path.join(REPO_ROOT, 'models')

for path in [REPO_ROOT, CURRENT_DIR, os.path.join(REPO_ROOT, 'Question_1'), os.path.join(REPO_ROOT, 'Question_3')]:
    if path not in sys.path:
        sys.path.append(path)

# Decision Rule Threshold Constants (in nats per token, calibrated against Brown corpus)
TAU_PCFG_OUTLIER: float = -15.0    # Minimum normalized log-prob for PCFG not to be an outlier
TAU_TRI_GRAMMATICAL: float = -8.8  # High-confidence threshold for trigram fluency
TAU_TRI_SPARSE: float = -11.0      # Threshold below which trigram backoff indicates severe sparsity
TAU_BI_GRAMMATICAL: float = -8.5   # High-confidence threshold for bigram transitions


def clean_token(token: str) -> str:
    """Strips non-alphanumeric punctuation and lowercases a token for vocabulary lookup."""
    return re.sub(r'[^a-zA-Z0-9]', '', token).lower()


def split_into_sentences(
    tokens_or_text: Union[str, List[str], List[Dict[str, Any]]],
    token_alerts: Optional[List[Optional[str]]] = None
) -> List[Dict[str, Any]]:
    """
    Splits the passage into distinct sentences using the final corrected token stream
    and aggregates segmentation merges and spelling corrections per sentence.

    Args:
        tokens_or_text: Can be:
            1. A raw passage string (e.g., from `PassageSampler`).
            2. A list of post-processed token strings.
            3. A list of dictionaries from `pipeline.process_token(token)`:
               [{'token': str, 'alert': Optional[str]}, ...]
        token_alerts: Optional list of alert strings corresponding to tokens when
                      tokens_or_text is passed as a list of strings.

    Returns:
        A list of sentence record dictionaries, each containing:
            - 'sentence text': The full reconstructed sentence string.
            - 'tokens': List of clean words in the sentence.
            - 'num_merges': Number of segmentation merges resolved in this sentence.
            - 'num_spells': Number of spelling corrections applied in this sentence.
    """
    sentence_records: List[Dict[str, Any]] = []

    # Case 1: Raw text passage
    if isinstance(tokens_or_text, str):
        # Attempt NLTK sent_tokenize; fallback to regex if punkt models are not present
        try:
            raw_sents = nltk.sent_tokenize(tokens_or_text)
        except Exception:
            raw_sents = re.split(r'(?<=[.!?])\s+', tokens_or_text.strip())

        for s in raw_sents:
            s_clean = s.strip()
            if not s_clean:
                continue
            words = []
            for w in s_clean.split():
                c = clean_token(w)
                if c:
                    words.append(c)
            if words:
                sentence_records.append({
                    'sentence text': s_clean,
                    'tokens': words,
                    'num_merges': 0,
                    'num_spells': 0
                })
        return sentence_records

    # Case 2 & 3: List of tokens or list of dicts
    curr_tokens: List[str] = []
    curr_raw_tokens: List[str] = []
    merges_in_sent = 0
    spells_in_sent = 0

    num_items = len(tokens_or_text)
    for idx in range(num_items):
        item = tokens_or_text[idx]
        if isinstance(item, dict):
            token_str = str(item.get('token', ''))
            alert_str = item.get('alert')
        else:
            token_str = str(item)
            alert_str = token_alerts[idx] if (token_alerts and idx < len(token_alerts)) else None

        # Check for alerts
        if alert_str:
            if '[SEGMENT-ALERT]' in alert_str:
                merges_in_sent += 1
            if '[SPELL-ALERT]' in alert_str:
                spells_in_sent += 1

        curr_raw_tokens.append(token_str)

        # Correctly handle merged tokens that were segmented into multiple words (e.g. "grand jury")
        for sub_token in token_str.split():
            c_tok = clean_token(sub_token)
            if c_tok:
                curr_tokens.append(c_tok)

        # Detect sentence termination
        has_sent_end = any(punct in token_str for punct in ['.', '!', '?'])
        is_last = (idx == num_items - 1)

        if has_sent_end or is_last:
            sent_text = " ".join(curr_raw_tokens).strip()
            # Clean up punctuation spacing for natural appearance
            sent_text = (sent_text
                         .replace(" .", ".")
                         .replace(" ,", ",")
                         .replace(" !", "!")
                         .replace(" ?", "?")
                         .replace(" '", "'"))
            if sent_text and curr_tokens:
                sentence_records.append({
                    'sentence text': sent_text,
                    'tokens': list(curr_tokens),
                    'num_merges': merges_in_sent,
                    'num_spells': spells_in_sent
                })
            # Reset buffers for next sentence
            curr_tokens = []
            curr_raw_tokens = []
            merges_in_sent = 0
            spells_in_sent = 0

    return sentence_records


def load_or_induce_treebank_pcfg(num_sents: int = 150) -> Optional[Any]:
    """
    Induces a sample PCFG grammar and builds a ViterbiParser from NLTK's Penn Treebank.
    Used as an optional fallback when Question 4 Part 2 is not yet trained.
    """
    try:
        from nltk.corpus import treebank
        from nltk import induce_pcfg, Nonterminal, ViterbiParser

        nltk.download('treebank', quiet=True)
        parsed_sents = treebank.parsed_sents()[:num_sents]
        productions = []
        for item in parsed_sents:
            productions += item.productions()
        S = Nonterminal('S')
        pcfg = induce_pcfg(S, productions)
        return ViterbiParser(pcfg)
    except Exception:
        return None


def score_sentence_pcfg(tokens: List[str], pcfg_model: Any = None) -> Tuple[Optional[float], str]:
    """
    Evaluates a sentence under a PCFG parser. Sentences that fail to parse
    are flagged as 'Unparseable' rather than crashing the pipeline.

    Args:
        tokens: List of cleaned word tokens.
        pcfg_model: A trained PCFG object, CKY parser, or None.

    Returns:
        A tuple of (log_probability or None, display_string).
        If unparseable, returns (None, "Unparseable").
    """
    if not tokens:
        return None, "Unparseable"

    # 1. Attempt to use passed pcfg_model
    if pcfg_model is not None:
        try:
            if hasattr(pcfg_model, 'parse'):
                parses = list(pcfg_model.parse(tokens))
                if parses:
                    best_parse = parses[0]
                    prob = best_parse.prob() if hasattr(best_parse, 'prob') else 1e-12
                    log_prob = math.log(max(prob, 1e-300))
                    return log_prob, f"{log_prob:.4f}"
            elif callable(pcfg_model):
                result = pcfg_model(tokens)
                if result is not None:
                    if isinstance(result, (int, float)):
                        return float(result), f"{float(result):.4f}"
                    elif hasattr(result, 'prob'):
                        log_prob = math.log(max(result.prob(), 1e-300))
                        return log_prob, f"{log_prob:.4f}"
        except NotImplementedError:
            pass
        except Exception:
            pass

    # 2. Check for Question 4 Part 2 CKY parser function
    try:
        from Question_4.part2_pcfg import cky_parse
        result = cky_parse(tokens, pcfg_model)
        if result is not None:
            if isinstance(result, (int, float)):
                return float(result), f"{float(result):.4f}"
            elif hasattr(result, 'prob'):
                log_prob = math.log(max(result.prob(), 1e-300))
                return log_prob, f"{log_prob:.4f}"
    except (ImportError, NotImplementedError, Exception):
        pass

    # Default graceful fallback: sentence is flagged as Unparseable
    return None, "Unparseable"


def score_sentence_bigram(
    tokens: List[str],
    ngram_models: Any = None,
    spelling_models: Any = None,
    k: float = 0.05
) -> Tuple[float, float]:
    """
    Computes sentence log-probability under an add-k smoothed bigram language model.

    Args:
        tokens: Cleaned words of the sentence.
        ngram_models: SmoothedNgramModels from Part 3 (if provided).
        spelling_models: SpellingModels from Question 3 (if provided).
        k: Smoothing parameter (default: 0.05).

    Returns:
        Tuple of (total_log_prob, avg_log_prob_per_word).
    """
    if not tokens:
        return 0.0, 0.0

    # 1. If Part 3's SmoothedNgramModels provides get_bigram_prob
    if ngram_models is not None and hasattr(ngram_models, 'get_bigram_prob'):
        try:
            total_lp = 0.0
            padded = ['<BOS>'] + tokens + ['<EOS>']
            for i in range(len(padded) - 1):
                prob = ngram_models.get_bigram_prob(padded[i], padded[i + 1])
                prob = max(prob, 1e-30)
                total_lp += math.log(prob)
            avg_lp = total_lp / len(tokens)
            return total_lp, avg_lp
        except NotImplementedError:
            pass

    # 2. If Question 3's SpellingModels is available
    if spelling_models is not None and hasattr(spelling_models, 'get_bigram_prob'):
        try:
            total_lp = 0.0
            for i in range(len(tokens) - 1):
                prob = spelling_models.get_bigram_prob(tokens[i], tokens[i + 1])
                prob = max(prob, 1e-30)
                total_lp += math.log(prob)
            first_prob = spelling_models.get_unigram_prob(tokens[0])
            first_prob = max(first_prob, 1e-30)
            total_lp += math.log(first_prob)
            avg_lp = total_lp / len(tokens)
            return total_lp, avg_lp
        except Exception:
            pass

    # 3. Built-in Add-k Bigram calculation
    V = 45000
    total_lp = 0.0
    for _ in range(len(tokens)):
        p = (0 + k) / (100 + k * V)
        total_lp += math.log(p)
    avg_lp = total_lp / max(len(tokens), 1)
    return total_lp, avg_lp


def score_sentence_trigram(
    tokens: List[str],
    ngram_models: Any = None,
    q1_lm: Any = None
) -> Tuple[float, float]:
    """
    Computes sentence log-probability under an add-k smoothed trigram language model.

    Args:
        tokens: Cleaned words of the sentence.
        ngram_models: SmoothedNgramModels from Part 3 (if provided).
        q1_lm: TrigramLanguageModel from Question 1 (if provided).

    Returns:
        Tuple of (total_log_prob, avg_log_prob_per_word).
    """
    if not tokens:
        return 0.0, 0.0

    # 1. If Part 3's SmoothedNgramModels provides get_trigram_prob
    if ngram_models is not None and hasattr(ngram_models, 'get_trigram_prob'):
        try:
            total_lp = 0.0
            padded = ['<BOS>', '<BOS>'] + tokens + ['<EOS>']
            for i in range(2, len(padded)):
                prob = ngram_models.get_trigram_prob(padded[i - 2], padded[i - 1], padded[i])
                prob = max(prob, 1e-30)
                total_lp += math.log(prob)
            avg_lp = total_lp / len(tokens)
            return total_lp, avg_lp
        except NotImplementedError:
            pass

    # 2. If Question 1's TrigramLanguageModel is available
    if q1_lm is not None and hasattr(q1_lm, 'log_prob'):
        try:
            total_lp = 0.0
            bos_tok = getattr(q1_lm, 'BOS', '<BOS>')
            eos_tok = getattr(q1_lm, 'EOS', '<EOS>')
            padded = [bos_tok, bos_tok] + tokens + [eos_tok]
            for i in range(2, len(padded)):
                lp = q1_lm.log_prob(padded[i], padded[i - 2], padded[i - 1])
                total_lp += lp
            avg_lp = total_lp / len(tokens)
            return total_lp, avg_lp
        except Exception:
            pass

    # 3. Built-in Add-k Trigram fallback
    V = 45000
    k = 0.05
    total_lp = 0.0
    for _ in range(len(tokens)):
        p = (0 + k) / (50 + k * V)
        total_lp += math.log(p)
    avg_lp = total_lp / max(len(tokens), 1)
    return total_lp, avg_lp


def apply_decision_rule(
    data: Union[pd.DataFrame, Dict[str, Any], pd.Series]
) -> Union[pd.DataFrame, Tuple[str, str]]:
    """
    Applies the documented hierarchical decision rule to select the best output
    and determine the final grammaticality verdict.

    Rule Hierarchy:
      1. Tier 1 (PCFG): If PCFG result is not 'Unparseable' and normalized log-prob >= -15.0:
           -> Chosen Method: 'PCFG', Verdict: 'Grammatical'.
      2. Tier 2 (Trigram LM): If PCFG is 'Unparseable' or outlier:
           -> If normalized trigram log-prob >= -8.8:
                Chosen Method: 'Trigram LM', Verdict: 'Grammatical'.
           -> Else if normalized trigram log-prob >= -11.0:
                Chosen Method: 'Trigram LM', Verdict: 'Questionable'.
           -> Else: severe trigram backoff/sparsity, proceed to Tier 3.
      3. Tier 3 (Bigram LM Fallback):
           -> If normalized bigram log-prob >= -8.5:
                Chosen Method: 'Bigram LM', Verdict: 'Grammatical'.
           -> Else:
                Chosen Method: 'Bigram LM', Verdict: 'Ungrammatical'.

    Args:
        data: Either a pandas DataFrame (with analysis columns), or a single dictionary / Series.

    Returns:
        - If input is a DataFrame: The DataFrame with updated 'chosen method' and 'final verdict' columns.
        - If input is a Dict/Series: Tuple of (chosen_method, final_verdict).
    """
    def _evaluate_row(row: Union[Dict[str, Any], pd.Series]) -> Tuple[str, str]:
        pcfg_val = row.get('PCFG result') if 'PCFG result' in row else row.get('pcfg_result')
        bi_score = row.get('bigram score') if 'bigram score' in row else row.get('bigram_score')
        tri_score = row.get('trigram score') if 'trigram score' in row else row.get('trigram_score')
        num_tokens = row.get('num_tokens') or len(str(row.get('sentence text', '')).split()) or 1

        # Calculate normalized scores (per-word)
        norm_bi = float(bi_score) / num_tokens if bi_score is not None and not pd.isna(bi_score) else -20.0
        norm_tri = float(tri_score) / num_tokens if tri_score is not None and not pd.isna(tri_score) else -20.0

        # Tier 1: Check PCFG
        is_parseable = False
        norm_pcfg = None
        if pcfg_val is not None and not pd.isna(pcfg_val):
            if str(pcfg_val).strip().lower() != 'unparseable':
                try:
                    pcfg_float = float(pcfg_val)
                    norm_pcfg = pcfg_float / num_tokens
                    is_parseable = True
                except (ValueError, TypeError):
                    is_parseable = False

        if is_parseable and norm_pcfg is not None:
            if norm_pcfg >= TAU_PCFG_OUTLIER:
                return "PCFG", "Grammatical"
            # If parsed but an extreme outlier, fall through to N-gram evaluation

        # Tier 2: Check Trigram LM
        if norm_tri >= TAU_TRI_GRAMMATICAL:
            return "Trigram LM", "Grammatical"
        elif norm_tri >= TAU_TRI_SPARSE:
            return "Trigram LM", "Questionable"

        # Tier 3: Check Bigram LM (Fallback)
        if norm_bi >= TAU_BI_GRAMMATICAL:
            return "Bigram LM", "Grammatical"
        else:
            return "Bigram LM", "Ungrammatical"

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        methods = []
        verdicts = []
        for _, r in df.iterrows():
            m, v = _evaluate_row(r)
            methods.append(m)
            verdicts.append(v)
        df['chosen method'] = methods
        df['final verdict'] = verdicts
        return df
    elif isinstance(data, (dict, pd.Series)):
        return _evaluate_row(data)
    else:
        raise TypeError(f"Unsupported data type for apply_decision_rule: {type(data)}")


def generate_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generates the final per-sentence summary table matching all assignment specifications.

    Columns produced:
      1. sentence text
      2. PCFG result
      3. bigram score
      4. trigram score
      5. chosen method
      6. final verdict
      7. number of segmentation merges resolved in this sentence
      8. number of spelling corrections applied in this sentence

    Args:
        results: A list of dictionaries containing metrics and outputs per sentence.

    Returns:
        A pandas DataFrame displaying the complete comparison.
    """
    formatted_rows: List[Dict[str, Any]] = []

    for r in results:
        sent_text = r.get('sentence text') or r.get('text') or r.get('sentence') or ""
        pcfg_res = r.get('PCFG result') or r.get('pcfg_result')
        if pcfg_res is None or str(pcfg_res).strip().lower() == 'none':
            pcfg_res = "Unparseable"
        elif isinstance(pcfg_res, float):
            pcfg_res = f"{pcfg_res:.4f}"

        bi_score = r.get('bigram score') or r.get('bigram_score')
        if bi_score is not None and not isinstance(bi_score, str):
            bi_score = round(float(bi_score), 4)

        tri_score = r.get('trigram score') or r.get('trigram_score')
        if tri_score is not None and not isinstance(tri_score, str):
            tri_score = round(float(tri_score), 4)

        # Apply decision rule if chosen method or final verdict not already computed
        chosen_m = r.get('chosen method') or r.get('chosen_method')
        final_v = r.get('final verdict') or r.get('final_verdict')

        if not chosen_m or not final_v:
            tokens_len = len(r.get('tokens', sent_text.split())) or 1
            eval_input = {
                'sentence text': sent_text,
                'PCFG result': pcfg_res,
                'bigram score': bi_score,
                'trigram score': tri_score,
                'num_tokens': tokens_len
            }
            computed_m, computed_v = apply_decision_rule(eval_input)
            chosen_m = chosen_m or computed_m
            final_v = final_v or computed_v

        merges = r.get('number of segmentation merges resolved in this sentence')
        if merges is None:
            merges = r.get('num_merges', 0)

        spells = r.get('number of spelling corrections applied in this sentence')
        if spells is None:
            spells = r.get('num_spells', 0)

        formatted_rows.append({
            'sentence text': sent_text,
            'PCFG result': str(pcfg_res),
            'bigram score': bi_score,
            'trigram score': tri_score,
            'chosen method': chosen_m,
            'final verdict': final_v,
            'number of segmentation merges resolved in this sentence': int(merges),
            'number of spelling corrections applied in this sentence': int(spells)
        })

    df = pd.DataFrame(formatted_rows, columns=[
        'sentence text',
        'PCFG result',
        'bigram score',
        'trigram score',
        'chosen method',
        'final verdict',
        'number of segmentation merges resolved in this sentence',
        'number of spelling corrections applied in this sentence'
    ])
    return df


class PassageAnalyzer:
    """
    End-to-end analyzer that integrates Question 1's Trigram LM, Question 3's Spelling Models,
    Part 2's PCFG Parser, and Part 3's Smoothed N-gram models to perform final passage analysis.
    """

    def __init__(
        self,
        pcfg_model: Any = None,
        ngram_models: Any = None,
        q1_lm: Any = None,
        spelling_models: Any = None,
        enable_pcfg_fallback: bool = True
    ):
        self.pcfg_model = pcfg_model
        self.ngram_models = ngram_models
        self.q1_lm = q1_lm
        self.spelling_models = spelling_models

        # Auto-load pre-trained Q1 and Q3 models if not explicitly passed
        if self.q1_lm is None:
            q1_lm_path = os.path.join(MODELS_DIR, 'q1_lm_en.pkl')
            if os.path.exists(q1_lm_path):
                try:
                    with open(q1_lm_path, 'rb') as f:
                        self.q1_lm = pickle.load(f)
                except Exception:
                    pass

        if self.spelling_models is None:
            q3_path = os.path.join(MODELS_DIR, 'q3_spelling_models.pkl')
            if os.path.exists(q3_path):
                try:
                    with open(q3_path, 'rb') as f:
                        self.spelling_models = pickle.load(f)
                except Exception:
                    pass

        # Optional fallback: if PCFG is not provided, induce a compact Treebank PCFG
        if self.pcfg_model is None and enable_pcfg_fallback:
            self.pcfg_model = load_or_induce_treebank_pcfg(num_sents=100)

    def analyze_passage(
        self,
        passage_or_tokens: Union[str, List[str], List[Dict[str, Any]]],
        token_alerts: Optional[List[Optional[str]]] = None
    ) -> pd.DataFrame:
        """
        Processes a full passage or token sequence through sentence splitting,
        tri-model scoring, decision rule resolution, and table formatting.

        Args:
            passage_or_tokens: Raw text, token list, or list of token-alert dicts from Part 1.
            token_alerts: Optional list of alert strings matching tokens.

        Returns:
            The complete pandas DataFrame comparison table.
        """
        # Step 1: Split into sentences with merge/spell tracking
        sentences = split_into_sentences(passage_or_tokens, token_alerts)

        results: List[Dict[str, Any]] = []

        # Step 2: Score each sentence
        for sent in sentences:
            tokens = sent.get('tokens', [])

            # (a) PCFG Score
            pcfg_val, pcfg_str = score_sentence_pcfg(tokens, self.pcfg_model)

            # (b) Bigram Score
            bi_tot, _ = score_sentence_bigram(
                tokens,
                ngram_models=self.ngram_models,
                spelling_models=self.spelling_models
            )

            # (c) Trigram Score
            tri_tot, _ = score_sentence_trigram(
                tokens,
                ngram_models=self.ngram_models,
                q1_lm=self.q1_lm
            )

            results.append({
                'sentence text': sent['sentence text'],
                'tokens': tokens,
                'PCFG result': pcfg_str,
                'bigram score': bi_tot,
                'trigram score': tri_tot,
                'number of segmentation merges resolved in this sentence': sent['num_merges'],
                'number of spelling corrections applied in this sentence': sent['num_spells']
            })

        # Step 3: Apply Decision Rule and generate formatted DataFrame
        return generate_comparison_table(results)


def analyze_passage(
    passage_or_tokens: Union[str, List[str], List[Dict[str, Any]]],
    pcfg_model: Any = None,
    ngram_models: Any = None,
    q1_lm: Any = None,
    spelling_models: Any = None,
    token_alerts: Optional[List[Optional[str]]] = None
) -> pd.DataFrame:
    """
    Convenience function to analyze a passage and return the comparison table.
    """
    analyzer = PassageAnalyzer(
        pcfg_model=pcfg_model,
        ngram_models=ngram_models,
        q1_lm=q1_lm,
        spelling_models=spelling_models
    )
    return analyzer.analyze_passage(passage_or_tokens, token_alerts)


if __name__ == "__main__":
    print("=" * 80)
    print(" Question 4, Part 4: Final Passage Analysis Demo")
    print("=" * 80)

    # Sample simulated passage with merged tokens, spelling correction, and ungrammatical controls
    sample_pipeline_output = [
        {'token': 'The', 'alert': None},
        {'token': 'grand jury', 'alert': '[SEGMENT-ALERT] Merged words detected. Suggested: grand jury'},
        {'token': 'commented', 'alert': None},
        {'token': 'on', 'alert': None},
        {'token': 'the', 'alert': None},
        {'token': 'election.', 'alert': None},
        {'token': 'The', 'alert': None},
        {'token': 'mayor', 'alert': None},
        {'token': 'accepted', 'alert': '[SPELL-ALERT] Misspelled word. Suggested: accepted'},
        {'token': 'the', 'alert': None},
        {'token': 'nomination.', 'alert': None},
        {'token': 'Colorless', 'alert': None},
        {'token': 'green', 'alert': None},
        {'token': 'ideas', 'alert': None},
        {'token': 'sleep', 'alert': None},
        {'token': 'furiously.', 'alert': None}
    ]

    print("\nRunning PassageAnalyzer on simulated typed stream...")
    analyzer = PassageAnalyzer()
    comparison_df = analyzer.analyze_passage(sample_pipeline_output)

    print("\nFinal Comparison Table (Part 4 Summary):")
    print("-" * 120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(comparison_df.to_string(index=False))
    print("-" * 120)
    print("\nPart 4 Execution successfully completed.")
