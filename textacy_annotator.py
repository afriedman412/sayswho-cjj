import spacy
import spacy_transformers
import textacy
from textacy import extract
from typing import List

class quoteAttributor:
    """
    Turns text into attributed quotes.

    Assumes text is pre-processed. (This is downstream of any soup processing!)

    TODO: Raise error (or warning?) if multiple clusters match.
    TODO: Quotation errors still happen, need to account:
    67CN-9C61-F03F-K4G3-00000-00
    """
    def __init__(self, diff: int=5):
        """
        So you don't have to initiate the spacy model every time.
        """
        self.nlp = spacy.load("en_coreference_web_trf")
        self.ner_nlp = spacy.load("en_core_web_sm")
        self.diff = diff
        return
    
    def parse_text(self, t: str):
        """ 
        Input: 
            t (string) - formatted text of an article
            
        Ouput:
            self.doc - spacy coref-parsed doc
            self.quotes - list of textacy-extracted quotes
        """
        self.doc = self.nlp(t)
        t_doc = textacy.make_spacy_doc(t, lang="en_core_web_sm")
        self.quotes = [q for q in extract.triples.direct_quotations(t_doc)]
        self.parse_entities(t)

    def parse_entities(self, t: str):
        """
        I was having trouble getting the coref model to return entities, so this uses a smaller spacy model to do that.

        Input:
            t (string) - formatted text of an article

        Output:
            self.ner_doc - spacy-parsed doc
            self.ent_index (dict) - {entity start: entity label}
        """
        self.ner_doc = self.ner_nlp(t)
        self.ent_index = {e.start:e for e in self.ner_doc.ents}
        
    def compare_quote_to_cluster_member(
            self, 
            q: extract.triples.DQTriple, 
            cluster_member: spacy.tokens.span.Span
        ): 
        """
        Compares the starting character of the quote speaker and the cluster member, as well as the quote speaker sentence and the cluster member sentence to determine equivalence.
        
        Input:
            q (quote triple) - one textacy quote triple
            cluster_member - one spacy-parsed entity cluster member
            
        Output:
            bool
        """
        return all([
                abs(q.speaker[0].sent.start_char - cluster_member.sent.start_char) < self.diff,
                abs(q.speaker[0].idx - cluster_member.start_char) < self.diff
            ])
        
    def compare_quote_to_cluster(
            self, 
            q: extract.triples.DQTriple, 
            cluster: List[spacy.tokens.span.Span]
        ):
        """
        Returns first match. Doesn't consider further matches. Is this a problem?
        """
        try:
            return next(
                n for n, m in enumerate(cluster) if self.compare_quote_to_cluster_member(q, m)
            )
        except StopIteration:
            return
    
    def quotes_to_clusters(self):
        matches = []
        for n, q in enumerate(self.quotes):
            for c in self.doc.spans:
                if c.startswith("coref_clusters"):
                    match_idx = self.compare_quote_to_cluster(q, self.doc.spans[c])
                    if match_idx:
                        matches.append((n, c.split("_")[-1], match_idx))
                    else:
                        matches.append((n, None, None))

        return matches
    
    def match_quotes(self):
        """
        Deal with it.
        """
        self.matches = self.quotes_to_clusters()
    
    def load_cluster(self, n):
        try:
            return self.doc.spans[f"coref_clusters_{n}"]
        except IndexError:
            return
    
    def format_cluster_span(self, span, min_length: int=3):
        """
        Filters out spans of less than "min_length" and finds the index and label of any entities in the span.

        Input:
            span - span from a cluster
            min_length (int) - minimum length of span to return (in characters, not tokens)

        Output:
            spans (list) - list of spans and, if applicable, index and label of any entities in the span.
        """
        if len(span.text) > min_length:
            spans = [span]
            for t in span:
                if t.i in self.ent_index:
                    spans.append((t.i, self.ent_index[t.i].label_))
            return spans
        else:
            return
        
    def format_cluster(self, cluster, min_length: int=3):
        """
        Runs format_cluster_span on all spans, removes duplicates and sorts by reverse length.

        Might be a problem here with removing duplicates early!
        """
        return [
            self.format_cluster_span(s, min_length) for s in sorted(
                list(set([
            span for span in cluster])
            ), key=lambda s: len(s.text), reverse=True)
            ]
    
    def parse_match(self, match: tuple):
        quote = self.quotes[match[0]]
        if match[1]:
            cluster = self.format_cluster(self.load_cluster(match[1])[match[2]])
        else:
            cluster = None
        return quote, cluster
    
    def prettify_match(self, match: tuple):
        quote = self.quotes[match[0]]
        cluster = self.load_cluster(match[1]) if match[1] else None
        cluster_match = cluster[match[2]] if cluster else None

        print("quote:", quote.content)
        print("speaker:", ' '.join([s.text for s in quote.speaker]))
        print("cluster match:", cluster_match)
        print("cluster match context:", cluster_match.sent)
        print("full cluster:", cluster)

    def prettify_matches(self):
        for m in self.matches:
            self.prettify_match(m)
            print("---")
    