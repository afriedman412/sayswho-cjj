"""
Entirely cribbed from Textacy, but with my upgrades.

Moved it here because it was easier than worrying about loading the right version of Textacy.
"""
from operator import attrgetter
from typing import Iterable
from spacy.tokens import Doc, Span, Token
from .constants import QUOTATION_MARK_PAIRS, _ACTIVE_SUBJ_DEPS
from .quote_helpers import (
    DQTriple, expand_noun, expand_verb, windower, filter_cue_candidates, filter_speaker_candidates
)

def direct_quotations(doc: Doc) -> Iterable[DQTriple]:
    """
    Extract direct quotations with an attributable speaker from a document
    using simple rules and patterns. Does not extract indirect or mixed quotations!

    Args:
        doc

    Yields:
        Next direct quotation in ``doc`` as a (speaker, cue, content) triple.

    Notes:
        Loosely inspired by Krestel, Bergler, Witte. "Minding the Source: Automatic
        Tagging of Reported Speech in Newspaper Articles".
    """
    # pairs up quotation-like characters based on acceptable start/end combos
    # see constants for more info
    qtok = [tok for tok in doc if tok.is_quote]
    qtok_pair_idxs = []
    for n, q in enumerate(qtok):
        if q.i not in [q_[1] for q_ in qtok_pair_idxs]:
            for q_ in qtok[n+1:]:
                if (ord(q.text), ord(q_.text)) in QUOTATION_MARK_PAIRS:
                    qtok_pair_idxs.append((q.i, q_.i))
                    break
    
    def filter_quote_tokens(tok):
        return any(
                qts_idx <= tok.i <= qte_idx for qts_idx, qte_idx in qtok_pair_idxs
            )

    for qtok_start_idx, qtok_end_idx in qtok_pair_idxs:
        content = doc[qtok_start_idx : qtok_end_idx + 1]
        cue = None
        speaker = None
        # filter quotations by content
        if (
            # quotations should have at least a couple tokens
            # excluding the first/last quotation mark tokens
            len(content) < 4
            # filter out titles of books and such, if possible
            or all(
                tok.is_title
                for tok in content
                # if tok.pos in {NOUN, PROPN}
                if not (tok.is_punct or tok.is_stop)
            )
        ):
            continue

        triple = None
        for n, window_sents in enumerate([
            windower(qtok_start_idx, qtok_end_idx, doc, True), 
            windower(qtok_start_idx, qtok_end_idx, doc)
        ]):
        # get candidate cue verbs in window
            cue_cands = [
                tok
                for sent in window_sents
                for tok in sent
                if not filter_quote_tokens(tok)
                and filter_cue_candidates(tok)
            ]
            # sort candidates by proximity to quote content
            cue_cands = sorted(
                cue_cands,
                key=lambda cc: min(abs(cc.i - qtok_start_idx), abs(cc.i - qtok_end_idx)),
            )
            for cue_cand in cue_cands:
                if cue is not None:
                    break
                speaker_cands = [
                    speaker_cand for speaker_cand in cue_cand.children
                    if not filter_quote_tokens(speaker_cand)
                    and filter_speaker_candidates(speaker_cand, qtok_start_idx, qtok_end_idx)
                ]
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
                break
