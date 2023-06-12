from collections import namedtuple
import regex as re
import os
from spacy.tokens import Doc, Span, Token
from typing import Iterable, Union, List
from .constants import (
    _reporting_verbs, _VERB_MODIFIER_DEPS, QUOTATION_MARK_PAIRS, 
    all_quotes, brack_regex, double_quotes, double_quotes_nospace_regex
    )
from spacy.symbols import VERB, PUNCT

DQTriple: tuple[list[Token], list[Token], Span] = namedtuple(
    "DQTriple", ["speaker", "cue", "content"]
)

def filter_cue_candidates(tok):
    return all([
        tok.pos == VERB,
        tok.lemma_ in _reporting_verbs
    ])

def filter_speaker_candidates(ch, i, j):
    return all([
            ch.pos!=PUNCT,
            ((ch.i >= i and ch.i >= j) or (ch.i <= i and ch.i <= j)),
        ])

def filter_quote_tokens(tok: Token, qtok_idx_pairs: List[tuple]) -> bool:
    return any(qts_idx <= tok.i <= qte_idx for qts_idx, qte_idx in qtok_idx_pairs)

def get_qtok_idx_pairs(input: Union[Doc, List[Token]]) -> List[tuple]:
    if isinstance(input, Doc):
        input = [tok for tok in input if tok.is_quote]
    qtok_idx_pairs = [(0,0)]
    for n, q in enumerate(input):
        if q.i not in [q_[1] for q_ in qtok_idx_pairs] and not bool(q.whitespace_) and q.i > qtok_idx_pairs[-1][1]:
            for q_ in input[n+1:]:
                if (ord(q.text), ord(q_.text)) in QUOTATION_MARK_PAIRS:
                    qtok_idx_pairs.append((q.i, q_.i))
                    break
    return qtok_idx_pairs

def expand_noun(tok: Token) -> list[Token]:
    """Expand a noun token to include all associated conjunct and compound nouns."""
    tok_and_conjuncts = [tok] + list(tok.conjuncts)
    compounds = [
        child
        for tc in tok_and_conjuncts
        for child in tc.children
        if child.dep_ == "compound"
    ]
    return tok_and_conjuncts + compounds

def expand_verb(tok: Token) -> list[Token]:
    """Expand a verb token to include all associated auxiliary and negation tokens."""
    verb_modifiers = [
        child for child in tok.children if child.dep in _VERB_MODIFIER_DEPS
    ]
    return [tok] + verb_modifiers

def line_break_window(i, j, doc):
    """
    Finds the boundaries of the paragraph containing doc[i:j].
    """
    for i_, j_ in list(zip(
        [tok.i for tok in doc if tok.text=="\n"],
        [tok.i for tok in doc if tok.text=="\n"][1:])
        ):
            if i_ <= i and j_ >= j:
                return (i_, j_)
    else:
        return (None, None)
    
def windower(i, j, doc, para=False) -> Iterable:
    """
    Two ways to search for cue and speaker: the old way, and a new way based on line breaks.
    """
    if para:
        i_, j_ = line_break_window(i, j, doc)
        if i_:
            return (
                sent 
                for sent in doc[i_+1:j_-1].sents
            )
        else:
            return []
    else:
        # get window of adjacent/overlapping sentences
        return (
            sent
            for sent in doc.sents
            # these boundary cases are a subtle bit of work...
            if (
                (sent.start < i and sent.end >= i - 1)
                or (sent.start <= j + 1 and sent.end > j)
            )
        )
    
def para_quote_fixer(p, exp: bool=False):
    if not p:
        return
    p = p.replace("\'\'", "\"")
    if exp:
        p = re.sub(r"(.{3,8}s\')(\s)", r"\1x\2", p)
    while re.search(double_quotes_nospace_regex, p):
        match = re.search(double_quotes_nospace_regex, p)
        if len(re.findall(brack_regex.format(double_quotes), p[:match.start()])) % 2 != 0:
            replacer = '" '
        else:
            replacer = ' "'
        p = p[:match.start()] + replacer + p[match.end():]
        if orphan_quote_finder(p):
            p += '"'
    return p.strip()

def orphan_quote_finder(p):
    if not (p[0] == "'" and p[-1] == "'") and p[0] in all_quotes and len(re.findall(brack_regex.format(double_quotes), p[1:])) % 2 == 0:
        return True
    return False

def prep_text_for_quote_detection(t, para_char="\n", exp: bool=False):
    return para_char.join([para_quote_fixer(p, exp=exp) for p in t.split(para_char) if p])