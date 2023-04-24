import spacy
import spacy_transformers
import textacy
from textacy import extract
from typing import List, Union
from tqdm import tqdm
from .article_helpers import get_json_data, full_parse

def compare_ents(e1, e2, min_diff=2):
    return all([abs(getattr(e1, attr)-getattr(e2,attr)) < min_diff for attr in ['start', 'end']])


class quoteAttributor:
    """
    Turns text into attributed quotes.

    Assumes text is pre-processed. (This is downstream of any soup processing!)

    TODO: Raise error (or warning?) if multiple clusters match.
    TODO: Quotation errors still happen, need to account:
    67CN-9C61-F03F-K4G3-00000-00
    """
    def __init__(
            self, 
            min_diff: int=5, 
            min_length=0,
            main_nlp_model="en_coreference_web_trf",
            ner_nlp_model="./ner_model/output/model-best",
            textacy_nlp_model="en_core_web_sm"
            ):
        """
        So you don't have to initiate the spacy model every time.

        Input:
            diff (int) - allowed distance between start characters/indexes in self.compare_quote_to_cluster
            min_length (int) - minimum length of span to return (in characters, not tokens) in self.format_cluster and self.format_cluster_span
        """
        self.nlp = spacy.load(main_nlp_model)
        self.ner_nlp = spacy.load(ner_nlp_model)
        self.ner_nlp.add_pipe("sentencizer")
        self.min_diff = min_diff
        self.min_length = min_length
        return
    
    def parse_text(self, t: str):
        """ 
        Input: 
            t (string) - formatted text of an article
            
        Ouput:
            self.doc - spacy coref-parsed doc
            self.quotes - list of textacy-extracted quotes

        TODO: do I need two spacy models?
        """
        # instantiate spacy doc
        self.doc = self.nlp(t) 

        # extract coref clusters
        self.clusters = {
            k.split("_")[-1]:v for k,v in self.doc.spans.items() if k.startswith("coref")
            } 
        
        # instantiate textacy spacy doc
        t_doc = textacy.make_spacy_doc(t, lang="en_core_web_sm") 

        # extract quotations
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
                abs(q.speaker[0].sent.start_char - cluster_member.sent.start_char) < self.min_diff,
                abs(q.speaker[0].idx - cluster_member.start_char) < self.min_diff
            ])
        
    def compare_quote_to_cluster(
            self, 
            q: extract.triples.DQTriple, 
            cluster: List[spacy.tokens.span.Span]
        ):
        """
        Finds first span in cluster that matches with provided quote.
        
        TODO: Doesn't consider further matches. Is this a problem?

        Input:
            q (quote) - textacy quote object
            cluster - coref cluster

        Output:
            n (int) or None - index of cluster member that matches quote (or None, if none match)
        """
        try:
            return next(
                n for n, m in enumerate(cluster) if self.compare_quote_to_cluster_member(q, m)
            )
        except StopIteration:
            return
    
    def quotes_to_clusters(self):
        """
        Iterates through quotes found in text, returning matches for each quote.

        Input:
            None

        Output:
            matches (tuple) - tuple of quote/cluster matches in the format tuple(quote index, cluster number, index of span in cluster) ...

            If no match is found, there will be an empty match (n, None None).
        """
        matches = []
        for n, q in enumerate(self.quotes):
            matched = False
            for k,v in self.clusters.items():
                match_idx = self.compare_quote_to_cluster(q, v)
                if match_idx:
                    matches.append((n, k, match_idx))
                    matched = True
            if not matched:
                matches.append((n, None, None))
        return matches
    
    def match_quotes(self):
        """
        Deal with it.
        """
        self.matches = self.quotes_to_clusters()
        self.get_ent_clusters()
    
    def load_cluster(self, n):
        """
        TODO: can I just do this with .get()?
        """
        try:
            return self.doc.spans[f"coref_clusters_{n}"]
        except KeyError:
            return
    
    def format_cluster_span(self, span):
        """
        Filters out spans of less than self.min_length and finds the index and label of any entities in the span.

        Input:
            span - span from a cluster

        Output:
            spans (list) - list of spans and, if applicable, index and label of any entities in the span.
        """
        if len(span.text) > self.min_length: # in characters, not tokens!
            spans = [span]
            for t in span:
                if t.i in self.ent_index:
                    spans.append((t.i, self.ent_index[t.i].label_))
            return spans
        else:
            return
        
    def format_cluster(self, cluster):
        """
        Runs format_cluster_span on all spans, removes duplicates and sorts by reverse length.

        Might be a problem here with removing duplicates early!

        Input:
            cluster - coref cluster

        Output:
            list of unique spans and any index/label of entities in the span, in the cluster
        """
        unique_spans = list(set([span for span in cluster]))
        sorted_spans = sorted(unique_spans, key=lambda s: len(s.text), reverse=True)
        return [self.format_cluster_span(s) for s in sorted_spans]
    
    def parse_match(self, match: Union[tuple, int]):
        """
        Input:
            match (tuple) - a quote/cluster match tuple (quote index, cluster number, index of span in cluster)
            
            OR

            match (int) - a match index

        Output:
            quote - textacy quote
            cluster - coref cluster
        """
        if isinstance(match, int):
            match = self.matches[match]

        quote = self.quotes[match[0]]
        if match[1]:
            cluster = self.clusters[match[1]]
            span_match = cluster[match[2]]
            cluster = [_ for _ in self.format_cluster(cluster) if _]
        else:
            cluster = None
            span_match = None
        return quote, cluster, span_match
    
    def get_ent_clusters(self):
        self.ent_clusters = []
        for c,v in self.clusters.items():
            for c_ in v:
                for e in self.ner_doc.ents:
                    if compare_ents(c_, e):
                        self.ent_clusters.append(c)
    
    # TODO: clean this up, now that there is a Match object
    
    def prettify_match(self, match: tuple):
        quote = self.quotes[match[0]]
        cluster = self.clusters[match[1]] if match[1] else None
        cluster_match = cluster[match[2]] if cluster else None

        print("quote:", quote.content)
        print("speaker:", ' '.join([s.text for s in quote.speaker]))

        if cluster_match:
            print("cluster match:", cluster_match)
            print("cluster match context:", cluster_match.sent)
            print("full cluster:", cluster)

    def prettify_matches(self):
        for m in self.matches:
            self.prettify_match(m)
            print("---")

    def write_match(self, match, f):
        quote, cluster, span_match = self.parse_match(match)
        f.write(f"content: {quote.content.text}, {quote.content.start_char}\n")
        f.write(f"speaker: {' '.join([s.text for s in quote.speaker])}, {quote.speaker[0].idx}\n")
        f.write(f"span match: {span_match if cluster else 'NONE'}\n")
        f.write(f"full cluster: {cluster if cluster else 'NONE'}\n")
        f.write("-----\n")

    def output_matches(self, ids, file_name, engine):
        for i in tqdm(ids):
            data = get_json_data(i, engine)
            t = full_parse(data)
            try:
                self.parse_text(t)
                self.match_quotes()
                with open(f"./{file_name}", "a+") as f:
                    f.write(f'**** {i} ****\n')
                    for match in self.matches:
                        self.write_match(match, f)
            except Exception as e:
                print(i, e.args)

    def make_match(self, match):
        quote_index, cluster_index, cluster_span = match
        new_match = Match(self, quote_index, cluster_index, cluster_span)
        if not new_match.cluster:
            new_match.compare_clusters(self.clusters)
        return new_match


class Match:
    """
    Automatically loads info for a quoteAttributor quote/cluster match.

    For easier referencing and QC and auditing of actual match.

    TODO: test speaker against all clusters if no cluster

    Input:
        qa (quoteAttributor) - qa that generated the match
        quote_index (int) - index of qa quote
        cluster_index (str) - index of qa cluster
            (from autogenerated "coref_cluster_N", so its functionally an int, but its used as a string so no need to convert)
        span_match (int) - index of cluster member match (aka "cluster_member" elsewhere)
    """
    def __init__(self, qa: quoteAttributor, quote_index, cluster_index=None, span_match=None):
        self.quote = qa.quotes[quote_index]
        self.content = self.quote.content.text
        self.quote_char = self.quote.content.start_char

        self.speaker = ' '.join([s.text for s in self.quote.speaker])
        self.speaker_index = self.quote.speaker[0].idx
        self.speaker_sent_char = self.quote.speaker[0].sent.start_char

        self.manual_speakers = None

        null_attrs = []

        if cluster_index:
            self.cluster_index = cluster_index
            if cluster_index in qa.ent_clusters:
                self.ent_cluster_match = "yes"
            self.cluster_full = qa.clusters[cluster_index]
            self.cluster = sorted(list(set([c.text for c in self.cluster_full])), key=lambda c: len(c), reverse=True)
            self.pred_speaker = self.cluster[0]

        else:
            null_attrs += ['cluster', 'pred_speaker', 'cluster_index', 'ent_cluster_match']

        if span_match:
            self.span_match = self.cluster_full[span_match]
            self.span_context = self.span_match.sent.text
            self.span_match_char = self.span_match.start_char
            self.span_match_sent_char = self.span_match.sent.start_char

        else:
            null_attrs += ['span_match', 'span_context', 'span_match_char', 'span_match_sent_char']
        
        for a in null_attrs:
            setattr(self, a, None)
        
        return
    
    def __repr__(self):
        return f"content: {self.content}\nspeaker: {self.speaker}"
    
    def compare_clusters(self, clusters):
        """
        If the match doesn't have a cluster, find any speakers in clusters that match manually.

        The idea here is if the speaker is "Rosenberg" to pull "Detective Jeff Rosenberg" from the clusters.

        quoteAttribution runs this automatically with its clusters if the match doesn't have a cluster!
        """
        self.manual_speakers = []
        for v in clusters.values():
            speaker_match = [v_ for v_ in v if self.speaker in v_.text.replace("@", "'")]
            if speaker_match:
                self.manual_speakers += speaker_match

        self.manual_speakers = list(set(self.manual_speakers))