"""
Rewritten with less overhead.
"""
import spacy
from typing import Union
from rapidfuzz import fuzz
from .attribution_helpers import (
    span_contains, 
    compare_spans, 
    compare_quote_to_cluster, 
    format_cluster, 
    compare_quote_to_cluster_member,
    quick_ent_analyzer,
    pronoun_check,
    filter_duplicate_ents,
    prune_cluster_people,
    clone_cluster
    )
from .quotes import direct_quotations
from .quote_helpers import DQTriple
from .constants import ent_like_words, ner_nlp, ent_like_words, QuoteEntMatch, EvalResults, QuoteClusterMatch

class Attributor:
    def __init__(
            self, 
            coref_nlp="en_coreference_web_trf",
            base_nlp="en_core_web_lg",
            ner_nlp=ner_nlp
            ):
        self.coref_nlp = spacy.load(coref_nlp)
        self.base_nlp = spacy.load(base_nlp)
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
            return sorted(
                list(
                set([(q.quote_index, q.ent_index) for q in self.ent_matches])
                ), key=lambda m: m[0]
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
        
    def attribute(self, t: str):
        """
        Top level function. Parses text, matches quotes to clusters and gets ent matches.

        TODO: Given that get_ent_matches accounts for self.ner, I can probably get rid of quotes_to_clusters and the if statement!

        Input:
            t (str) - text file to be analyzed and attributed
        """
        self.parse_text(t)
        self.quotes_to_clusters()
        if self.ner:
            self.get_ent_matches()

    def parse_text(self, t: str):
        """ 
        Imports text, gets coref clusters, copies coref clusters, finds PERSONS and gets NER matches.

        Input: 
            t (string) - formatted text of an article
            
        Ouput:
            self.coref_doc - spacy coref-parsed doc
            self.doc - spacy doc with coref clusters
            self.clusters - coref clusters
            self.quotes - list of textacy-extracted quotes
            self.persons - list of PERSON entities
            self.ner_doc - spacy doc with NER matches
        """
        # instantiate spacy doc
        self.coref_doc = self.coref_nlp(t)
        self.doc = self.base_nlp(t)

        # extract quotations
        self.quotes = [q for q in direct_quotations(self.doc)]

        # extract coref clusters and clone to doc
        self.clusters = {
            int(k.split("_")[-1]): clone_cluster(cluster, self.doc) 
            for k, cluster in self.coref_doc.spans.items() 
            if k.startswith("coref")
            }
        
        self.persons = [e for e in self.doc.ents if e.label_=="PERSON"]

        if self.ner:
            self.ner_doc = self.ner_nlp(t)
            self.ner_doc.ents = filter_duplicate_ents(self.ner_doc.ents)
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
    
    def get_ent_matches(self):
        """
        Makes all "pair" lists, then runs quick_ent_analyzer to get ent_matches.

        This exists because I wanted a step between macro_ent_finder and the results for easier testing.
        """
        quote_person_pairs, quote_cluster_pairs, quote_ent_pairs, cluster_ent_pairs, cluster_person_pairs, person_ent_pairs = self.macro_ent_finder()
        self.ent_matches = quick_ent_analyzer(
                        quote_person_pairs, 
                        quote_cluster_pairs, 
                        quote_ent_pairs,
                        cluster_ent_pairs,
                        cluster_person_pairs,
                        person_ent_pairs
                                        )

    def macro_ent_finder(self, prune=True):
        """
        Messy but lite ent finder. Easier than keeping track of all the ways to match ent and cluster.

        Can do 

        TODO: Track spans and methods?
        TODO: Ensure pronouns aren't being skipped!
        TODO: Make ratio threshold a variable
        """
        quote_person_pairs = []
        quote_ent_pairs = []
        quote_cluster_pairs = []
        for quote_index, quote in enumerate(self.quotes):
            if self.ner:
                if any([' '.join([s.text for s in quote.speaker]).lower() == elw for elw in ent_like_words]):
                    quote_ent_pairs.append((quote_index, None))
                for ent_index, ent in enumerate(self.ents):
                    if span_contains(quote, ent):
                        quote_ent_pairs.append((quote_index, ent_index))
            for person_index, person in enumerate(self.persons):
                if span_contains(quote, person):
                    quote_person_pairs.append((quote_index, person_index))

            for cluster_index, cluster in self.clusters.items():
                if prune:
                    cluster = prune_cluster_people(cluster)
                for span_index, span in enumerate(cluster):
                    if compare_quote_to_cluster_member(quote, span):
                        quote_cluster_pairs.append((quote_index, cluster_index))
                        
        cluster_ent_pairs = []
        cluster_person_pairs = []
        for cluster_index, cluster in self.clusters.items():
            if prune:
                    cluster = prune_cluster_people(cluster)
            for span_index, span in enumerate(cluster):
                if not pronoun_check(span):
                    if self.ner:
                        for ent_index, ent in enumerate(self.ents):
                            if compare_spans(span, ent) or fuzz.partial_ratio(span.text, ent.text) > 95:
                                cluster_ent_pairs.append((cluster_index, ent_index))
                    for person_index, person in enumerate(self.persons):
                        if span_contains(person, span):
                            cluster_person_pairs.append((cluster_index, person_index))
                    
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
        
def evaluate(a: Attributor):
    """
    Used to score results.

    number of quotes, number of quotes from ents, unique ents quoted 
    """
    return EvalResults(
        len(a.quotes),
        len(set([m[0] for m in list(a.reduce_ent_matches)])),
        len(set([m[1] for m in list(a.reduce_ent_matches)]))
    )

def get_ent_score(a: Attributor):
    if not a.ner:
        return
    else:
        return (
            len(a.quotes), 
            [m.quote_index for m in a.ner_matches if m.contains_ent], 
            len(set([m.ent_method for m in a.ner_matches if m.contains_ent]))
        )
