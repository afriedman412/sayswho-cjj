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
    """
    def __init__(self, diff: int=5):
        """
        So you don't have to initiate the spacy model every time.
        """
        self.nlp = spacy.load("en_coreference_web_trf")
        self.diff = diff
        return
    
    def parse_text(self, t: str):
        """ 
        Input: 
            t (string) - formatted text of an article
            
        Ouput:
            self.quotes - list of textacy-extracted quotes
            self.clusters - spacy-parsed entity clusters
        """
        self.doc = self.nlp(t)
        t_doc = textacy.make_spacy_doc(t, lang='en_core_web_sm')
        self.quotes = [q for q in extract.triples.direct_quotations(t_doc)]
        
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
        Returns first match. Doesn't consider further matches. Is s this a problem?
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
        return matches
    
    def match_quotes(self):
        """
        Deal with it.
        """
        self.matches = self.quotes_to_clusters()
    
    def load_cluster(self, n):
        return self.doc.spans[f"coref_clusters_{n}"]
        
    def parse_match(self, match: tuple):
        return self.quotes[match[0]], self.load_cluster(match[1])[match[2]]
    
    def prettify_match(self, match: tuple):
        quote = self.quotes[match[0]]
        cluster = self.load_cluster(match[1])
        cluster_match = cluster[match[2]]

        print("quote:", quote.content)
        print("speaker:", quote.speaker[0])
        print("cluster match:", cluster_match)
        print("cluster match context:", cluster_match.sent)
        print("full cluster:", cluster)

    def prettify_matches(self):
        for m in self.matches:
            self.prettify_match(m)
            print("---")
    