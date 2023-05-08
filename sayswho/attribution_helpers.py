from spacy.tokens.span import Span
from spacy.tokens.token import Token
from spacy.tokens import SpanGroup
from .textacy_boot_helpers import DQTriple
from collections import namedtuple
from typing import Union

LilMatch: tuple[int, Union[int, None], Union[int, None], bool, str] = namedtuple(
    "LilMatch", 
    ["quote_index", "cluster_index", "span_index", "contains_ent", "ent_method"], 
    defaults=(0, None, None, False, None)
)

ClusterEnt: tuple[int, int, Span, int] = namedtuple(
    "ClusterEnt",
    ["cluster_index", "span_index", "span", "ent_start"]
)

Boundaries: tuple[int, int] = namedtuple(
    "boundaries",
    ['start', 'end']
)

class Match:
    """
    Automatically loads info for a quoteAttributor quote/cluster match.

    For easier referencing and QC and auditing of actual match.

    This probably doesn't need to be its own class eventually.

    TODO: test speaker against all clusters if no cluster

    Input:
        qa (quoteAttributor) - qa that generated the match
        lil_match - a LilMatch tuple (quote_index, cluster_index, contains_ent, ent_method)
    """
    def __init__(self, qa, lil_match: LilMatch):
        self.quote = qa.quotes[lil_match.quote_index]
        self.content = self.quote.content.text
        self.quote_char = self.quote.content.start_char

        # speaker is the canonical speaker from textacy
        self.speaker = ' '.join([s.text for s in self.quote.speaker])
        self.speaker_index = self.quote.speaker[0].idx
        self.speaker_sent_char = self.quote.speaker[0].sent.start_char

        self.manual_speakers = None

        null_attrs = []

        self.cluster_index = lil_match.cluster_index
        self.contains_ent = lil_match.contains_ent
        self.ent_method = lil_match.ent_method

        if lil_match.cluster_index:
            self.cluster = qa.clusters[lil_match.cluster_index]
            # pred_speaker is the longest span in the matched cluster
            # TODO: you can probably do better!
            self.pred_speaker = sorted(list(set([c.text for c in self.cluster])), key=lambda c: len(c), reverse=True)[0]

        else:
            null_attrs += ['cluster', 'pred_speaker']
            self.compare_clusters(qa.clusters)

        if lil_match.cluster_index:
            self.span_match = self.cluster[lil_match.span_index]
            self.span_context = self.span_match.sent.text
            self.span_match_char = self.span_match.start_char
            self.span_match_sent_char = self.span_match.sent.start_char

        else:
            null_attrs += ['span_match', 'span_context', 'span_match_char', 'span_match_sent_char']
        
        for a in null_attrs:
            setattr(self, a, None)
        
        return
    
    def __repr__(self):
        return '\n'.join([f"{k}: {self.__dict__.get(k)}" for k in [
            "content", "speaker", "pred_speaker", "manual_speakers_", "contains_ent", "ent_method"
        ]])
    
    def compare_clusters(self, clusters):
        """
        If the match doesn't have a cluster, find any speakers in clusters that match manually.

        The idea here is if the speaker is "Rosenberg" to pull "Detective Jeff Rosenberg" from the clusters.

        quoteAttribution runs this automatically with its clusters if the match doesn't have a cluster!

        Is this even in use?
        """
        self.manual_speakers = []
        for span in clusters.values():
            speaker_match = [v_ for v_ in span if self.speaker in v_.text.replace("@", "'")]
            if speaker_match:
                self.manual_speakers += speaker_match

        self.manual_speakers_ = list(set([m.text for m in self.manual_speakers]))

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

def compare_spans(
            s1: Span, 
            s2: Span,
            min_entity_diff: int
            ) -> bool:
        """
        Compares two spans to see if their starts and ends are less than self.min_entity_diff.

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
    Runs format_cluster_span on all spans, removes duplicates and sorts by reverse length.

    TODO: is there any reason to not remove duplicates here?

    Input:
        cluster - coref cluster

    Output:
        list of unique spans in the cluster
    """
    unique_spans = list(set([span for span in cluster]))
    sorted_spans = sorted(unique_spans, key=lambda s: len(s.text), reverse=True)
    return [s for s in sorted_spans if len(s.text) > min_length]

def compare_quote_to_cluster_member(
        quote: DQTriple,
        span: Span,
        min_speaker_diff: int
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

def compare_quote_to_cluster(
        quote: DQTriple, 
        cluster: SpanGroup,
        min_speaker_diff: int
    ):
    """
    Finds first span in cluster that matches (according to self.compare_quote_to_cluster_member) with provided quote.
    
    TODO: Doesn't consider further matches. Is this a problem?

    Input:
        quote - textacy quote object
        cluster - coref cluster

    Output:
        cluster_index (int) or None - index of cluster member that matches quote (or None, if none match)
    """
    try:
        return next(
            cluster_index for cluster_index, cluster_member in enumerate(cluster) \
                if compare_quote_to_cluster_member(quote, cluster_member, min_speaker_diff)
        )
    except StopIteration:
        return -1