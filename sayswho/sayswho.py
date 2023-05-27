"""
Rewritten with less overhead.
"""
import spacy
from typing import Union
from rapidfuzz import fuzz
import numpy as np
from .attribution_helpers import (
    span_contains, 
    compare_spans, 
    format_cluster, 
    compare_quote_to_cluster_member,
    pronoun_check,
    filter_duplicate_ents,
    prune_cluster_people,
    clone_cluster,
    get_manual_speaker_cluster
    )
from .quotes import direct_quotations
from .quote_helpers import DQTriple
from .constants import ent_like_words, ner_nlp, ent_like_words, QuoteEntMatch, QuoteClusterMatch, EvalResults

class Attributor:
    """
    TODO: Add manual speaker matching
    """
    def __init__(
            self, 
            coref_nlp="en_coreference_web_trf",
            base_nlp="en_core_web_lg",
            ner_nlp=ner_nlp,
            prune=True
            ):
        self.coref_nlp = spacy.load(coref_nlp)
        self.base_nlp = spacy.load(base_nlp)
        if ner_nlp:
            self.ner_nlp = spacy.load(ner_nlp)
            self.ner_nlp.add_pipe("sentencizer")
        self.prune = prune

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
        self.get_matches()

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
        if self.prune:
            self.clusters = {n:prune_cluster_people(cluster) for n, cluster in self.clusters.items()}
        
        self.persons = [e for e in self.doc.ents if e.label_=="PERSON"]

        if self.ner:
            self.ner_doc = self.ner_nlp(t)
            self.ner_doc.ents = filter_duplicate_ents(self.ner_doc.ents)
        return
    
    def get_matches(self):
        """
        Makes all "pair" lists, then runs quick_ent_analyzer to get ent_matches.

        This exists because I wanted a step between macro_ent_finder and the results for easier testing.
        """
        pairs_dicto = self.macro_ent_finder()
        self.make_matches(pairs_dicto)
        
    def get_manual_quote_ent_pairs(self) -> list:
        return [
            QuoteEntMatch(quote_index)
            for quote_index, quote in enumerate(self.quotes)
            if any([' '.join([s.text for s in quote.speaker]).lower() == elw for elw in ent_like_words])
            ]
    
    def get_manual_quote_cluster_pairs(self, quote_cluster_pairs):
        return [
                (quote_index, cluster_index) 
                for quote_index, quote in enumerate(self.quotes)
                if quote_index not in [m[0] for m in set(quote_cluster_pairs)]
                if quote.speaker[0].text in [p.text for p in self.persons]
                for cluster_index, cluster in self.clusters.items()
                if get_manual_speaker_cluster(quote, cluster)
            ]

    def macro_ent_finder(self) -> dict:
        """
        Messy but lite ent finder. Easier than keeping track of all the ways to match ent and cluster.

        TODO: Ensure pronouns aren't being skipped!
        TODO: Make ratio threshold a variable
        """
        pairs_dicto = {p:[] for p in [
            'quotes_persons', 'quotes_ents', 'quotes_clusters', 
            'clusters_ents', 'clusters_persons', 'persons_ents'
            ]}

        for quote_index, quote in enumerate(self.quotes):
            if self.ner:
                pairs_dicto['quotes_ents'] += [
                        (quote_index, ent_index)
                        for ent_index, ent in enumerate(self.ents)
                        if span_contains(quote, ent)
                        ]
            pairs_dicto['quotes_persons'] += [
                    (quote_index, person_index)
                    for person_index, person in enumerate(self.persons)
                    if span_contains(quote, person)
                    ]
            pairs_dicto['quotes_clusters'] += [
                    (quote_index, cluster_index) 
                    for cluster_index, cluster in self.clusters.items()
                    for span in cluster
                    if compare_quote_to_cluster_member(quote, span)
                    ]
                        
        for cluster_index, cluster in self.clusters.items():
            for span in cluster:
                if not pronoun_check(span):
                    if self.ner:
                        pairs_dicto['clusters_ents'] += [
                            (cluster_index, ent_index)
                            for ent_index, ent in enumerate(self.ents)
                            if compare_spans(span, ent) 
                            or fuzz.partial_ratio(span.text, ent.text) > 95
                            ]
                    pairs_dicto['clusters_persons'] += [
                        (cluster_index, person_index)
                        for person_index, person in enumerate(self.persons)
                        if span_contains(person, span)
                        ]
        pairs_dicto['persons_ents'] += [
            (person_index, ent_index)
            for person_index, p in enumerate(self.persons)
            for ent_index, ent in enumerate(self.ents)
            if span_contains(p, ent)
            ]
        
        pairs_dicto['quotes_clusters'] += self.get_manual_quote_cluster_pairs(pairs_dicto['quotes_clusters'])
        return pairs_dicto
    
    def make_matrix(self, key, pairs):
        x, y = key.split("_")
        m = np.zeros([len(self.__getattribute__(_)) for _ in [x,y]])
        for i,j in pairs:
            m[i,j] = 1
        return m
    
    def make_matches(self, pairs_dicto: dict):
        arrays = {k: self.make_matrix(k, v) for k,v in pairs_dicto.items()}

        self.ent_matches = [
            QuoteEntMatch(quote_index=i, ent_index=j) for i,j in np.concatenate(
                [
                    np.transpose(np.nonzero(arrays['quotes_ents'])),
                    np.transpose(np.nonzero(arrays['quotes_clusters'].dot(arrays['clusters_ents']))),
                    np.transpose(np.nonzero(arrays['quotes_persons'].dot(arrays['persons_ents']))),
                    np.transpose(
                        np.nonzero(
                            arrays['quotes_clusters'].dot(arrays['clusters_persons'].dot(arrays['persons_ents']))
                        )
                    )
                ]
            )]
        
        self.ent_matches = sorted(
            list(set(self.ent_matches + self.get_manual_quote_ent_pairs())),
            key=lambda m: m.quote_index
            )

        self.quote_matches = sorted(
            list(set([
            QuoteClusterMatch(i, j) for i,j in np.concatenate(
                (np.transpose(np.nonzero(arrays['quotes_persons'].dot(arrays['clusters_persons'].T))),
                 np.transpose(np.nonzero(arrays['quotes_clusters'])))
            )])), key=lambda m: m.quote_index
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
