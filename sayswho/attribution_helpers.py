from spacy.tokens import Span, SpanGroup, Token
from .textacy_boot_helpers import DQTriple
from collections import namedtuple
from typing import Union
from spacy.symbols import PRON
from .constants import min_entity_diff, min_speaker_diff, min_length

QuoteClusterMatch: tuple[int, Union[int, None], Union[int, None], bool, str] = namedtuple(
    "QuoteClusterMatch", 
    ["quote_index", "cluster_index", "span_index", "contains_ent", "ent_method"], 
    defaults=(0, None, None, False, None)
)

ClusterEntMatch: tuple[int, int, Span, int] = namedtuple(
    "ClusterEntMatch",
    ["cluster_index", "span_index", "span", "ent_start"]
)

Boundaries: tuple[int, int] = namedtuple(
    "boundaries",
    ['start', 'end']
)

def get_boundaries(t: Union[Token, Span]) -> Boundaries:
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
    
    else:
        raise TypeError("input needs to be Token or Span!")

def text_contains(
        t1: Union[Span, Token],
        t2: Union[Span, Token]
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

def format_cluster(cluster, min_length: int):
    """
    Removes cluster duplicates and sorts cluster by reverse length.

    TODO: is there any reason to not remove duplicates here?

    Input:
        cluster - coref cluster

    Output:
        list of unique spans in the cluster
    """
    unique_spans = list(set([span for span in cluster]))
    sorted_spans = sorted(unique_spans, key=lambda s: len(s.text), reverse=True)
    return [s for s in sorted_spans if len(s.text) > min_length]
    
def get_manual_speakers(quote, clusters):
    """
    If the match doesn't have a cluster, find any speakers in clusters that match manually.

    The idea here is if the speaker is "Rosenberg" to pull "Detective Jeff Rosenberg" from the clusters.

    Attributor runs this automatically with its clusters if the match doesn't have a cluster!
    """
    if any([
            len(quote.speaker) > 1,
            quote.speaker[0].pos_ != PRON
        ]):
        manual_speakers = []
        speaker = ' '.join([s.text for s in quote.speaker])
        for span in clusters.values():
            for v_ in span:
                if speaker in v_.text.replace("@", "'"):
                    manual_speakers.append(v_)
        return manual_speakers
    else:
        return
    
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
            abs(getattr(s1, attr)-getattr(s2,attr)) < min_entity_diff for attr in ['start', 'end']
        ])