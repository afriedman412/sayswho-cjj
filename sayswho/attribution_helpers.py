from spacy.tokens import Span, SpanGroup, Token
from .quote_helpers import DQTriple
from collections import namedtuple
from typing import Union
from spacy.symbols import PRON
from .constants import min_entity_diff, min_speaker_diff, min_length

QuoteClusterMatch: tuple[int, int, int, int] = namedtuple(
    "QuoteClusterMatch", 
    ["quote_index", "cluster_index", "span_index"], 
    defaults=(None, None, None)
)

QuoteEntMatch: tuple[int, int, int, int] = namedtuple(
    "QuoteEntMatch", 
    ["quote_index", "cluster_index", "person_index", "ent_index"],
    defaults=(None, None, None, None)
)

ClusterEntMatch: tuple[int, int, Span, int] = namedtuple(
    "ClusterEntMatch",
    ["cluster_index", "span_index", "span", "ent_start"]
)

Boundaries: tuple[int, int] = namedtuple(
    "boundaries",
    ['start', 'end']
)

def get_boundaries(t: Union[Token, Span, list]) -> Boundaries:
    """
    Convenience function that returns a Boundaries tuple with the start and the end character of t.

    Necessary because Token and Span have differently named attributes.

    Input:
        t (Token or Span) - spacy token or span object

    Output:
        Boundares(start, end) with start character and end character of t.
    """
    if isinstance(t, Token):
        return Boundaries(t.idx, t.idx+len(t))
    
    if isinstance(t, Span):
        return Boundaries(t.start_char, t.end_char)
    
    if isinstance(t, DQTriple):
        return Boundaries(
            get_boundaries(t.speaker[0]).start,
            get_boundaries(t.speaker[-1]).end,
            )
    
    else:
        raise TypeError("input needs to be Token or Span!")
    
def get_text(t: Union[Token, Span, list]) -> str:
    """
    Convenience function, because quote speakers are lists of tokens.
    """
    if isinstance(t, Token) or isinstance(t, Span):
        return t.text
    if isinstance(t, list):
        return ' '.join([t_.text for t_ in t])

def span_contains(
        t1: Union[Span, Token, DQTriple],
        t2: Union[Span, Token, DQTriple]
) -> bool:
    """
    Does t1 contain t2 or v/v? Assumes both are from the same doc!!

    Uses get_boundaries beacuse quote speakers are tokens but entities are spans.

    TODO: verify same doc

    Input:
        t1 and t2 - spacy spans or tokens (from the same doc)

    Output:
        bool - whether either t1 contains the other t2

    """
    b1, b2 = tuple([get_boundaries(t) for t in [t1, t2]])

    if (b1.start <= b2.start and b1.end >= b2.end) or (b2.start <= b1.start and b2.end >= b1.end):
        return True
    
    else:
        return False

def format_cluster(cluster, min_length: int=min_length):
    """
    Removes cluster duplicates and sorts cluster by reverse length.

    TODO: is there any reason to not remove duplicates here?
    TODO: remove pronouns

    Input:
        cluster - coref cluster

    Output:
        list of unique spans in the cluster
    """
    unique_spans = list(set([span for span in cluster]))
    sorted_spans = sorted(unique_spans, key=lambda s: len(s.text), reverse=True)
    return [s for s in sorted_spans if len(s.text) > min_length]

def pronoun_check(t, doc=None):
    if len(t) == 1:
        if doc:
            return doc[t[0].i].pos_ == PRON
        else:
            return t[0].pos_ == PRON
    return False
    
def get_manual_speaker_cluster(quote, cluster):
    """
    If the match doesn't have a cluster, find any speakers in clusters that match manually.

    The idea here is if the speaker is "Rosenberg" to pull "Detective Jeff Rosenberg" from the clusters.

    Attributor runs this automatically with its clusters if the match doesn't have a cluster!
    """
    if any([
            len(quote.speaker) > 1,
            quote.speaker[0].pos_ != PRON
        ]):
        speaker = ' '.join([s.text for s in quote.speaker])
        for span in cluster:
            if speaker in span.text.replace("@", "'"):
                return True
    else:
        return False
    
def compare_quote_to_cluster(
        quote: DQTriple, 
        cluster: SpanGroup,
    ):
    """
    Finds first span in cluster that matches (according to compare_quote_to_cluster_member) with provided quote.
    
    TODO: Doesn't consider further matches. Is this a problem?

    Input:
        quote - textacy quote object
        cluster - coref cluster

    Output:
        cluster_index (int) - index of cluster member that matches quote (or -1, if none match)
    """
    try:
        return next(
            cluster_index for cluster_index, cluster_member 
            in enumerate(cluster)
            if compare_quote_to_cluster_member(quote, cluster_member)
        )
    except StopIteration:
        return -1
    
def compare_quote_to_cluster_member(
        quote: DQTriple,
        span: Span
    ): 
    """
    Compares the starting character of the quote speaker and the cluster member as well as the quote speaker sentence and the cluster member sentence to determine equivalence.
    
    Input:
        q (quote triple) - one textacy quote triple
        cluster_member - one spacy-parsed entity cluster member
        
    Output:
        bool
    """
    if abs(quote.speaker[0].sent.start_char - span.sent.start_char) < min_speaker_diff:
        if abs(quote.speaker[0].idx - span.start_char) < min_speaker_diff:
            return True
    if span.start_char <= quote.speaker[0].idx:
        if span.end_char >= (quote.speaker[-1].idx + len(quote.speaker[-1])):
            return True
    return False

def compare_spans(
        s1: Span, 
        s2: Span,
        ) -> bool:
    """
    Compares two spans to see if their starts and ends are less than min_entity_diff.

    If compare

    Input:
        s1 and s2 - spacy spans
        min_entity_diff (int) - threshold for difference in start and ends
    Output:
        bool - whether the two spans start and end close enough to each other to be "equivalent"

    """
    return all([
            abs(getattr(s1, attr)-getattr(s2, attr)) < min_entity_diff for attr in ['start', 'end']
        ])

def quick_ent_analyzer(
        quote_person_pairs, 
        quote_cluster_pairs, 
        quote_ent_pairs,
        cluster_ent_pairs,
        cluster_person_pairs,
        person_ent_pairs
        ):
    ent_matches = []
    for qp in quote_person_pairs:
        for pe in person_ent_pairs:
            if qp[1] == pe[0]:
                ent_matches.append(QuoteEntMatch(qp[0], None, pe[0], pe[1]))

    for qc in quote_cluster_pairs:
        for ce in cluster_ent_pairs:
            if qc[1] == ce[0]:
                ent_matches.append(QuoteEntMatch(qc[0], ce[0], None, ce[1]))
        for cp in cluster_person_pairs:
            if qc[1] == cp[0]:
                for pe in person_ent_pairs:
                    if cp[1] == pe[0]:
                        ent_matches.append(QuoteEntMatch(qc[0], cp[0], pe[0], pe[1]))

    for qc in quote_cluster_pairs:
        if qc[1] == None:
            ent_matches.append(QuoteEntMatch(qc[0]))

    for qe in quote_ent_pairs:
        ent_matches.append(QuoteEntMatch(qe[0], None, None, qe[1]))
    return set(ent_matches)