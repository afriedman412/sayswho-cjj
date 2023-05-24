from .sayswho import Attributor
from spacy.tokens import Doc
from .constants import QUOTATION_MARK_PAIRS, _ACTIVE_SUBJ_DEPS
from .quote_helpers import (
    windower, filter_cue_candidates, filter_quote_tokens, 
    filter_speaker_candidates, expand_noun, expand_verb, DQTriple
    )
from operator import attrgetter

class QuoteFinder:
    def __init__(self, doc):
        self.doc = doc
        self.qtok = get_qtok(self.doc)
        self.qpairs = get_qpairs(self.qtok)

    def get_quotes(self, filter_cues=True, filter_speakers=True):
        self.quotes = [self.get_triple(qpair, filter_cues, filter_speakers) for qpair in self.qpairs]

    def get_cue_candidates(self, window_sents, filter_=True):
        if filter_:
            return [
                    tok
                    for sent in window_sents
                    for tok in sent
                    if filter_cue_candidates(tok)
                    and not filter_quote_tokens(tok, self.qpairs)
                ]
        else:
             return [
                    tok
                    for sent in window_sents
                    for tok in sent
                    if filter_cue_candidates(tok)
                ]
    
    def get_speaker_candidates(self, cue_candidate, i, j, filter_=True):
        if filter_:
            return [
                    speaker_cand for speaker_cand in cue_candidate.children
                    if not filter_quote_tokens(speaker_cand, self.qpairs)
                    and filter_speaker_candidates(speaker_cand, i, j)
                ]
        else:
            return [
                    speaker_cand for speaker_cand in cue_candidate.children
                    if filter_speaker_candidates(speaker_cand, i, j)
                ]
    

    def get_triple(self, qtok_pair, filter_cues=True, filter_speakers=True):
        i, j = qtok_pair
        content = self.doc[i:j+1]
        cue = None
        speaker = None
        for window_sents in [
            windower(i, j, self.doc, True), windower(i, j, self.doc)
        ]:
            cue_candidates = sorted(
                self.get_cue_candidates(window_sents, filter_cues),
                key=lambda cc: min(abs(cc.i - i), abs(cc.i - j))
            )

            for cue_cand in cue_candidates:
                if cue is not None:
                    break
                speaker_cands = self.get_speaker_candidates(cue_cand, i, j, filter_speakers)
                for speaker_cand in speaker_cands:
                    if speaker_cand.dep in _ACTIVE_SUBJ_DEPS:
                        cue = expand_verb(cue_cand)
                        speaker = expand_noun(speaker_cand)
                        break
                if content and cue and speaker:
                    return DQTriple(
                        speaker=sorted(speaker, key=attrgetter("i")),
                        cue=sorted(cue, key=attrgetter("i")),
                        content=content,
                    )

def get_qtok(doc: Doc):
    return [tok for tok in doc if tok.is_quote]

def get_qpairs(qtok: list):
    qtok_pair_idxs = []
    for n, q in enumerate(qtok):
        if not bool(q.whitespace_) and q.i not in [q_[1] for q_ in qtok_pair_idxs]:
            for q_ in qtok[n+1:]:
                try:
                    if (ord(q.text), ord(q_.text)) in QUOTATION_MARK_PAIRS:
                        qtok_pair_idxs.append((q.i, q_.i))
                        break
                except TypeError:
                    print(q.text, q_.text)
    return qtok_pair_idxs

def get_triple(qtok_pair, doc, qtok_pairs):
    qtok_start_idx, qtok_end_idx = qtok_pair
    content = doc[qtok_start_idx : qtok_end_idx + 1]
    cue = None
    speaker = None
    for n, window_sents in enumerate([
            windower(qtok_start_idx, qtok_end_idx, doc, True), 
            windower(qtok_start_idx, qtok_end_idx, doc)
        ]):
        # get candidate cue verbs in window
            cue_cands = [
                tok for sent in window_sents for tok in sent \
                if not filter_quote_tokens(tok, qtok_pairs) and filter_cue_candidates(tok)
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
            if not filter_quote_tokens(speaker_cand, qtok_pairs)
            and filter_speaker_candidates(speaker_cand, qtok_start_idx, qtok_end_idx)
        ]
        for speaker_cand in speaker_cands:
            if speaker_cand.dep in _ACTIVE_SUBJ_DEPS:
                cue = expand_verb(cue_cand)
                speaker = expand_noun(speaker_cand)
                break
    if content and cue and speaker:
        return DQTriple(
            speaker=sorted(speaker, key=attrgetter("i")),
            cue=sorted(cue, key=attrgetter("i")),
            content=content,
        )
    