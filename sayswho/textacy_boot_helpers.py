import collections
from spacy.tokens import Doc, Span, Token
from typing import Iterable
from .constants import _reporting_verbs, _VERB_MODIFIER_DEPS
from spacy.symbols import VERB, PUNCT

DQTriple: tuple[list[Token], list[Token], Span] = collections.namedtuple(
    "DQTriple", ["speaker", "cue", "content"]
)

def filter_cue_candidates(tok):
    return all([
        tok.pos == VERB,
        tok.lemma_ in _reporting_verbs
    ])

def get_cue_candidates(window_sents):
    return [
            tok
            for sent in window_sents
            for tok in sent
            if filter_cue_candidates(tok)
        ]

def filter_speaker_candidates(ch, i, j):
    return all([
            ch.pos!=PUNCT,
            ((ch.i >= i and ch.i >= j) or (ch.i <= i and ch.i <= j)),
        ])

def expand_noun(tok: Token) -> list[Token]:
    """Expand a noun token to include all associated conjunct and compound nouns."""
    tok_and_conjuncts = [tok] + list(tok.conjuncts)
    compounds = [
        child
        for tc in tok_and_conjuncts
        for child in tc.children
        # TODO: why doesn't compound import from spacy.symbols?
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
    for i_, j_ in list(zip(
        [tok.i for tok in doc if tok.text=="\n"],
        [tok.i for tok in doc if tok.text=="\n"][1:])
        ):
            if i_ <= i and j_ >= j:
                return (i_, j_)
    else:
        return (None, None)
    
def windower(i, j, doc, para=False) -> Iterable:
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