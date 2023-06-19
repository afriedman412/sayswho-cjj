from .quote_helpers import (
    old_windower, windower, expand_noun, expand_verb, DQTriple
    )
from .constants import _ACTIVE_SUBJ_DEPS, _reporting_verbs, min_quote_length, QUOTATION_MARK_PAIRS
from spacy.tokens import Doc, Token
from spacy.symbols import VERB, PUNCT
from operator import attrgetter
import regex as re

def direct_quotations(doc: Doc, exp: bool=False):
    qtoks = [tok for tok in doc if tok.is_quote or (re.match(r"(\n)+", tok.text))]
    qtok_idx_pairs = [(-1,-1)]
    for n, q in enumerate(qtoks):
        if (
            not bool(q.whitespace_)
            and q.i not in [q_[1] for q_ in qtok_idx_pairs] 
            and q.i > qtok_idx_pairs[-1][1]
            ):
            for q_ in qtoks[n+1:]:
                if (ord(q.text), ord(q_.text)) in QUOTATION_MARK_PAIRS:
                    qtok_idx_pairs.append((q.i, q_.i))
                    break  
    qtok_idx_pairs = qtok_idx_pairs[1:]

    def filter_quote_tokens(tok: Token) -> bool:
        return any(i <= tok.i <= j for i, j in qtok_idx_pairs)

    for i, j in qtok_idx_pairs:
        content = doc[i:j]
        if (
            len(content.text.split()) <= min_quote_length
            or all (
                tok.is_title
                for tok in content
                if not (tok.is_punct or tok.is_stop)           
                )
            ):
            continue
        cue = None
        speaker = None

        windy = [windower(content, "overlap"), windower(content, "linebreaks")]
        for window_sents in windy:
            cue_candidates = [
                    tok
                    for sent in window_sents
                    for tok in sent
                    if tok.pos == VERB
                    and tok.lemma_ in _reporting_verbs
                    and not filter_quote_tokens(tok)
                ]
            cue_candidates = sorted(
                cue_candidates,
                key=lambda cc: min(abs(cc.i - i), abs(cc.i - j))
            )
            
            for cue_cand in cue_candidates:
                if cue is not None:
                    break
                speaker_cands = [
                    speaker_cand for speaker_cand in cue_cand.children
                    if speaker_cand.pos != PUNCT
                    and ((speaker_cand.i >= j)
                    or (speaker_cand.i <= i))
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
                        content=doc[i:j+1],
                    )