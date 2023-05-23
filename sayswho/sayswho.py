"""
Rewritten with less overhead.
"""

import spacy
from spacy.symbols import PRON
from typing import Union
from fuzzywuzzy import fuzz
from .attribution_helpers import (
    QuoteClusterMatch,
    QuoteEntMatch,
    ClusterEntMatch,
    span_contains, 
    compare_spans, 
    compare_quote_to_cluster, 
    format_cluster, 
    compare_quote_to_cluster_member,
    quick_ent_analyzer,
    pronoun_check
    )
from .quotes import direct_quotations
from .quote_helpers import DQTriple
from .constants import ent_like_words, ner_nlp, min_length, ent_like_words

class Attributor:
    def __init__(
            self, 
            coref_nlp="en_coreference_web_trf",
            base_nlp="en_core_web_sm",
            ner_nlp=ner_nlp
            ):
        self.coref_nlp = spacy.load(coref_nlp)
        base_nlp = spacy.load(base_nlp)
        for p in ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'][::-1]:
            try:
                self.coref_nlp.add_pipe(p, source=base_nlp, first=True)
            except ValueError:
                print('problem adding pipe to coref nlp')
                continue
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
    
    @property
    def reduce_ent_matches(self):
        if 'ent_matches' in self.__dict__:
            return set([(q.quote_index, q.ent_index) for q in self.ent_matches])
        
    def attribute(self, t):
        self.parse_text(t)
        self.quotes_to_clusters()
        if self.ner:
            self.get_ent_matches()

    def get_ent_score(self):
        if not self.ner:
            return
        else:
            return (
                len(self.quotes), 
                [m.quote_index for m in self.ner_matches if m.contains_ent], 
                len(set([m.ent_method for m in self.ner_matches if m.contains_ent]))
            )
        
    def expand_match(self, match: Union[QuoteEntMatch, int]):
        if isinstance(match, int):
            match = self.ent_matches[match]
        for m_ in ['quote', 'cluster', 'person', 'ent']:
            if getattr(match, f"{m_}_index", None) is not None:
                i = self.__getattribute__(f"{m_}s")
                v = getattr(match, f"{m_}_index")
                data = format_cluster(i[v]) if m_=="cluster" else i[v]
                print(m_.upper(), f": {v}""\n", data, "\n")
        
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
        self.doc = self.coref_nlp(t) 

        # extract coref clusters
        self.clusters = {
            int(k.split("_")[-1]):v 
            for k,v in self.doc.spans.items() 
            if k.startswith("coref")
            } 

        self.persons = [e for e in self.doc.ents if e.label_=="PERSON"]
        
        # extract quotations
        self.quotes = [q for q in direct_quotations(self.doc)]

        if self.ner:
            self.ner_doc = self.ner_nlp(t)
            self.get_ent_clusters()
        return
    
    def quotes_to_clusters(self):
        self.quote_matches = [
            match 
            for quote_index, quote in enumerate(self.quotes)
            for match in [
                qc for qc in self.quote_to_clusters(quote, quote_index)
            ] 
        ]
        
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
                yield QuoteClusterMatch(quote_index, cluster_index, match_index)
                    
    def compare_quote_to_ents(
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
        # not using whole text because checking token boundaries
        if any([span_contains(quote.speaker[0], ent) for ent in self.ents]):
            return True, "contain"
        elif (cluster_index) and (cluster_index in [e.cluster_index for e in self.ent_clusters]):
            return True, f"cluster_{cluster_index}"
        elif ' '.join([tok.text for tok in quote.speaker]) in ent_like_words:
            return True, 'ent_like'
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
    
    def get_ent_matches(self):
        quote_person_pairs, quote_cluster_pairs, quote_ent_pairs, cluster_ent_pairs, cluster_person_pairs, person_ent_pairs = self.macro_ent_finder()
        self.ent_matches = quick_ent_analyzer(
                        quote_person_pairs, 
                        quote_cluster_pairs, 
                        quote_ent_pairs,
                        cluster_ent_pairs,
                        cluster_person_pairs,
                        person_ent_pairs
                                        )


    def macro_ent_finder(self):
        """
        Messy but lite ent finder. Easier than keeping track of all the ways to match ent and cluster.

        TODO: Track spans and methods?
        """
        quote_person_pairs = []
        quote_ent_pairs = []
        quote_cluster_pairs = []
        for quote_index, quote in enumerate(self.quotes):
            if any([' '.join([s.text for s in quote.speaker]).lower() == elw for elw in ent_like_words]):
                quote_cluster_pairs.append((quote_index, None, 1))
            for person_index, person in enumerate(self.persons):
                if span_contains(quote, person):
                    quote_person_pairs.append((quote_index, person_index, 2))
            for ent_index, ent in enumerate(self.ents):
                if span_contains(quote, ent):
                    quote_ent_pairs.append((quote_index, ent_index, 3))
            for cluster_index, cluster in self.clusters.items():
                for span_index, span in enumerate(cluster):
                    if compare_quote_to_cluster_member(quote, span):
                        quote_cluster_pairs.append((quote_index, cluster_index, 4))
                        
        cluster_ent_pairs = []
        cluster_person_pairs = []
        for cluster_index, cluster in self.clusters.items():
            for span_index, span in enumerate(cluster):
                if not pronoun_check(span):
                    for ent_index, ent in enumerate(self.ents):
                        if compare_spans(span, ent) or fuzz.partial_ratio(span.text, ent.text) > 95:
                            cluster_ent_pairs.append((cluster_index, ent_index, 5))
                    for person_index, person in enumerate(self.persons):
                        if span_contains(person, span):
                            cluster_person_pairs.append((cluster_index, person_index, 6))
                    
        person_ent_pairs = []
        for person_index, p in enumerate(self.persons):
            for ent_index, ent in enumerate(self.ents):
                if span_contains(p, ent):
                    person_ent_pairs.append((person_index, ent_index, 7))

        return (
            quote_person_pairs, 
            quote_cluster_pairs, 
            quote_ent_pairs,
            cluster_ent_pairs,
            cluster_person_pairs,
            person_ent_pairs
            )
        



