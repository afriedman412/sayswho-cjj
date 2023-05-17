"""
Rewritten with less overhead.
"""

import spacy
from spacy.tokens import SpanGroup, Span
from typing import Union
from .attribution_helpers import QuoteClusterMatch, ClusterEntMatch, text_contains, compare_spans, compare_quote_to_cluster_member, compare_quote_to_cluster
from .quotes import direct_quotations
from .quote_helpers import DQTriple

class Attributor:
    """
    TODO: add PERSON matching
    """
    def __init__(
            self, 
            nlp="en_coreference_web_trf",
            t_nlp="en_core_web_sm",
            ner_nlp=None
            ):
        self.nlp = spacy.load(nlp)
        self.t_nlp_model = t_nlp
        self.t_nlp = spacy.load(t_nlp)
        if ner_nlp:
            self.ner_nlp = spacy.load(ner_nlp)
            self.ner_nlp.add_pipe("sentencizer")

    @property
    def ner(self):
        """
        Whether to do NER processing.

        Output:
            bool
        """
        return 'ner_nlp' in self.__dict__
    
    @property
    def ents(self):
        """
        Returns entities (if NER)
        """
        if self.ner:
            return self.ner_doc.ents
        
    def get_cluster(self, cluster_index: Union[int, str]):
        """
        Convenience function to a coref cluster using only its index.

        Input:
            cluster_index (int) - the index of the cluster you want

        Output:
            the cluster
        """
        try:
            return self.doc.spans[f"coref_clusters_{cluster_index}"]
        except KeyError:
            return

    def attribute(self, t):
        self.parse_text(t)
        self.quotes_to_clusters()

    def parse_text(self, t: str):
        """ 
        Imports text, gets clusters, quotes and entities

        Input: 
            t (string) - formatted text of an article
            
        Ouput:
            self.doc - spacy coref-parsed doc
            self.quotes - list of textacy-extracted quotes

        TODO: do I need two spacy models? (or three?)
        """
        # instantiate spacy doc
        self.doc = self.nlp(t) 

        # extract coref clusters
        self.clusters = {
            int(k.split("_")[-1]):v 
            for k,v in self.doc.spans.items() 
            if k.startswith("coref")
            } 
        
        # instantiate textacy spacy doc
        t_doc = self.t_nlp(t) 

        # extract quotations
        self.quotes = [q for q in direct_quotations(t_doc)]

        if self.ner:
            # this was self.extract_entities, but easier to just do here
            self.ner_doc = self.ner_nlp(t)
            self.get_ent_clusters()

        return
    
    def quotes_to_clusters(self):
        self.matches = []
        for quote_index, quote in enumerate(self.quotes):
            self.matches += [qc for qc in self.quote_to_clusters(quote, quote_index)]
        
    def quote_to_clusters(
            self, 
            quote: DQTriple, 
            quote_index: int
            ):
        """
        Finds all matching clusters for quote

        Also evalutes each quote/cluster_index match for entity matches.

        Input:
            quote
            quote index
            cluster

        Output:
            matches (tuple) - tuple of quote/cluster matches in the format tuple(quote index, cluster number, index of span in cluster) ...

            If no match is found, there will be an empty match (n, None None).
        """
        for cluster_index, cluster in self.clusters.items():
            match_index = compare_quote_to_cluster(quote, cluster)
            if match_index > -1:
                if self.ner:
                    contains_ent, ent_method = self.quote_to_ents(quote, cluster_index)
                    yield QuoteClusterMatch(quote_index, cluster_index, match_index, contains_ent, ent_method)
                else:
                    yield QuoteClusterMatch(quote_index, cluster_index, match_index)
    
    def quote_to_ents(
            self, 
            quote: DQTriple, 
            cluster_index: Union[int, None]=None
            ) -> tuple[bool, str]:
        """
        Input:
            quote - the quote
            cluster_index (optional int) - cluster index

        Output:
            bool - whether quote speaker is or is within an entity, or whether the quote cluster contains an entity
            str - method of ent relationship
        """
        if any([text_contains(quote.speaker[0], ent) for ent in self.ents]):
            return True, "contain"
        elif (cluster_index) and (cluster_index in [e.cluster_index for e in self.ent_clusters]):
            return True, "cluster"
        else:
            return False, "none"
        
    def get_ent_clusters(self):
        """
        Compares ents to each token in a cluster, looks for match by token start distance and sentence start distance.

        Returns cluster/ent matches.

        Input:
            None

        Output:
            self.ent_clusters - list(ClusterEnt)
        """
        self.ent_clusters = []
        for cluster_index, cluster in self.clusters.items():
            for span_index, span in enumerate(cluster):
                for ent in self.ents:
                    if compare_spans(span, ent):
                        self.ent_clusters.append(
                            ClusterEntMatch(cluster_index, span_index, span, ent.start)
                        )
        return
    
    def expand_match(self, match: Union[int,QuoteClusterMatch]):
        if isinstance(int, match):
            match = self.matches[int]
        print(match.quote_index)
        print(self.quotes[match.quote_index])
        print(self.clusters[match.cluster_index])
        print(match.contains_ent)



