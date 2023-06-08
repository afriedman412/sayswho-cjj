"""
Entirely cribbed from Textacy, but with my upgrades.

Moved it here because it was easier than worrying about loading the right version of Textacy.

Consolidated into a class because it was easier than passing variables across functions.
"""
from spacy.tokens import Doc
from .constants import _ACTIVE_SUBJ_DEPS, min_quote_length
from .quote_helpers import (
    windower, filter_cue_candidates, filter_quote_tokens, 
    filter_speaker_candidates, expand_noun, expand_verb, 
    get_qtok, get_qpairs, DQTriple
    )
from operator import attrgetter

def direct_quotations(doc: Doc):
    qtok = get_qtok(doc)
    qpairs = [(i,j) for i,j in get_qpairs(qtok) if j-i > min_quote_length]

    for i, j in qpairs:
        content = doc[i:j+1]
        cue = None
        speaker = None

        for window_sents in [
            windower(i, j, doc, True), windower(i, j, doc)
        ]:
            cue_candidates = sorted(
                get_cue_candidates(window_sents, qpairs),
                key=lambda cc: min(abs(cc.i - i), abs(cc.i - j))
            )

            for cue_cand in cue_candidates:
                if cue is not None:
                    break
                speaker_cands = get_speaker_candidates(cue_cand, i, j, qpairs)
                for speaker_cand in speaker_cands:
                    if speaker_cand.dep in _ACTIVE_SUBJ_DEPS:
                        cue = expand_verb(cue_cand)
                        speaker = expand_noun(speaker_cand)
                        break
                if content and cue and speaker:
                    yield DQTriple(
                        speaker=sorted(speaker, key=attrgetter("i")),
                        cue=sorted(cue, key=attrgetter("i")),
                        content=content,
                    )
                
def get_cue_candidates(window_sents: list, qpairs, filter_: bool=True):
    if filter_:
        return [
                tok
                for sent in window_sents
                for tok in sent
                if filter_cue_candidates(tok)
                and not filter_quote_tokens(tok, qpairs)
            ]
    else:
            return [
                tok
                for sent in window_sents
                for tok in sent
                if filter_cue_candidates(tok)
            ]

def get_speaker_candidates(cue_candidate, i: int, j: int, qpairs, filter_: bool=True):
    if filter_:
        return [
                speaker_cand for speaker_cand in cue_candidate.children
                if not filter_quote_tokens(speaker_cand, qpairs)
                and filter_speaker_candidates(speaker_cand, i, j)
            ]
    else:
        return [
                speaker_cand for speaker_cand in cue_candidate.children
                if filter_speaker_candidates(speaker_cand, i, j)
            ] 
