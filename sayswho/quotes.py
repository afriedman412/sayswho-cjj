"""
Entirely cribbed from Textacy, but with my upgrades.

Moved it here because it was easier than worrying about loading the right version of Textacy.

TODO: Figure out multi-paragraph quotes of the format: ... "kdfslfsda \n "jilfdsaijfsadlij "jiadfsijasd"
Currently prep_text_for_quote_detection just adds a quotation mark on the end of any paragraph that starts with a quotation mark, doesn't end with one and contains an odd number of question marks. Not scientific!
"""
from .quote_helpers import (
    windower, filter_quote_tokens, expand_noun, expand_verb, 
    get_qtok_idx_pairs, DQTriple
    )
from .constants import _ACTIVE_SUBJ_DEPS, _reporting_verbs, min_quote_length
from spacy.tokens import Doc
from spacy.symbols import VERB, PUNCT
from operator import attrgetter

def direct_quotations(doc: Doc):
    qtok_idx_pairs = [(i,j) for i,j in get_qtok_idx_pairs(doc) if j-i > min_quote_length]

    for i, j in qtok_idx_pairs:
        content = doc[i:j+1]
        cue = None
        speaker = None

        for window_sents in [
            windower(i, j, doc, True), windower(i, j, doc)
        ]:
            cue_candidates = [
                    tok
                    for sent in window_sents
                    for tok in sent
                    if tok.pos == VERB
                    and tok.lemma_ in _reporting_verbs
                    and not filter_quote_tokens(tok, qtok_idx_pairs)
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
                        content=content,
                    )
