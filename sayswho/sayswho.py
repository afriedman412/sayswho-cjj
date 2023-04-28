"""
The bulk of the code for quote attribution.

TODO: extract canonical speaker for each cluster
TODO: add displacy / other viz
"""

import spacy
import spacy_transformers
import textacy
from textacy import extract
from textacy.extract.triples import DQTriple
from typing import List, Union
from .attribution_helpers import (
    LilMatch, ClusterEnt, Match, compare_quote_to_cluster, compare_spans, text_contains
)

class quoteAttributor:
    """
    Turns text into attributed quotes.

    Assumes text is pre-processed. (This is downstream of any soup processing!)

    TODO: Raise error (or warning?) if multiple clusters match.
    TODO: Quotation errors still happen, need to account:
    67CN-9C61-F03F-K4G3-00000-00
    672R-9X81-DY8S-B2WD-00000-00
    5W66-KYX1-DYNS-32VT-00000-00

    """
    def __init__(
            self, 
            min_speaker_diff: int=5, 
            min_entity_diff: int=2,
            min_length=0,
            main_nlp_model="en_coreference_web_trf",
            textacy_nlp_model="en_core_web_sm",
            ner_nlp_model=None
            
            ):
        """
        So you don't have to initiate the spacy model every time.

        Input:
            min_speaker_diff (int) - allowed distance between start characters/indexes in compare_quote_to_cluster
            min_entity_diff (int) - allowed distance between start/end of entity and span
            min_length (int) - minimum length of span to return (in characters, not tokens) in self.format_cluster and self.format_cluster_span
        """
        self.nlp = spacy.load(main_nlp_model)
        if ner_nlp_model:
            self.ner_nlp = spacy.load(ner_nlp_model)
            self.ner_nlp.add_pipe("sentencizer")
        self.min_speaker_diff = min_speaker_diff
        self.min_entity_diff = min_entity_diff
        self.min_length = min_length
        self.textacy_nlp_model = textacy_nlp_model # its not used until later
        return
    
    def attribute(self, t: str):
        self.parse_text(t)

        if self.ner:
            self.extract_entities(t)
            self.get_ent_clusters()
        
        self.quotes_to_clusters()

        return
    
    @property
    def ner(self):
        """
        Whether to do NER processing.

        Output:
            bool
        """
        return 'ner_nlp' in self.__dict__
    
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
        
    def expand_match(self, lil_match: Union[LilMatch, int]):
        """
        Converts a LilMatch into a Match
        TODO: redo now that there are Match objects -- do I need this function?

        Input:
            match (LilMatch) - a quote/cluster match tuple (quote index, cluster number, index of span in cluster)
            
            OR

            match (int) - a match index

        Output:
            big_match - a Match object
        """
        if isinstance(lil_match, int):
            lil_match = self.matches[lil_match]

        return Match(self, lil_match)
    
    def expand_matches(self):
        self.big_matches = [self.expand_match(m) for m in self.matches]
    
    def parse_text(self, t: str):
        """ 
        Imports text, gets clusters, quotes and entities

        TODO: do we need to save doc?

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
            int(k.split("_")[-1]):v for k,v in self.doc.spans.items() if k.startswith("coref")
            } 
        
        # instantiate textacy spacy doc
        t_doc = textacy.make_spacy_doc(t, lang=self.textacy_nlp_model) 

        # extract quotations
        self.quotes = [q for q in extract.triples.direct_quotations(t_doc)] 
        return

    def extract_entities(self, t: str):
        """
        Get any entities in text.

        Input:
            t (string) - formatted text of an article

        Output:
            self.ner_doc
            self.ents
        """
        self.ner_doc = self.ner_nlp(t)
        self.ents = self.ner_doc.ents
        return
    
    def get_ent_clusters(self):
        """
        Create list of cluster/entity matches.

        Input:
            None

        Output:
            self.ent_clusters - list(ClusterEnt)
        """
        self.ent_clusters = []
        for cluster_index, cluster in self.clusters.items():
            for span_index, span in enumerate(cluster):
                for ent in self.ents:
                    if compare_spans(span, ent, self.min_entity_diff):
                        self.ent_clusters.append(
                            ClusterEnt(cluster_index, span_index, span, ent.start)
                        )
        return
    
    def quote_to_ents(
            self, 
            quote: DQTriple, 
            cluster_index: Union[int, None]=None
            ) -> bool:
        """
        Input:
            quote - the quote
            cluster_index (optional int) - cluster index

        Output:
            bool - whether quote speaker is or is within an entity, or whether the quote cluster contains an entity
        """
        if any([text_contains(quote.speaker[0], ent) for ent in self.ents]):
            return True
        if cluster_index and (cluster_index in [e.cluster_index for e in self.ent_clusters]):
            return True
        else:
            return False

    def quotes_to_clusters(self):
        """
        Iterates through quotes found in text, returning matches for each quote.

        Also evalutes each quote/cluster_index match for entity matches.

        Input:
            None

        Output:
            matches (tuple) - tuple of quote/cluster matches in the format tuple(quote index, cluster number, index of span in cluster) ...

            If no match is found, there will be an empty match (n, None None).
        """
        self.matches = []
        for quote_index, quote in enumerate(self.quotes): # for each quote...
            matched = False
            for cluster_index, cluster in self.clusters.items(): # for each cluster...
                match_index = compare_quote_to_cluster(quote, cluster, self.min_speaker_diff)
                if match_index > -1: # if the speaker matches any of the spans in the cluster ...
                    # if self.ner: # check if the cluster contains an entity ...
                    #     contains_ent = self.quote_to_ents(quote, cluster_index)
                    # append
                    self.matches.append(
                        LilMatch(
                        quote_index, 
                        cluster_index, 
                        match_index, 
                        self.quote_to_ents(quote, cluster_index)
                        ))
                    matched = True

            # if not matched yet ...
            if not matched:
                # if self.ner and self.quote_to_ents(quote): # check for speaker/entity matches...
                #     self.matches.append(LilMatch(quote_index, None, None, True))
                # else: # or return a blank match
                self.matches.append(
                    LilMatch(
                    quote_index, 
                    None, 
                    None,
                    self.quote_to_ents(quote))
                    )
        return self.matches