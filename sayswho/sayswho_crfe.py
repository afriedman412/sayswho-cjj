import spacy
from spacy.tokens import Doc, Span, Token
from typing import Union
from .textacy_bootleg import direct_quotations

class CRFE:
    def __init__(self, nlp='en_core_web_trf', ner_nlp="output/model-last/"):
        self.nlp = spacy.load(nlp)
        self.nlp.add_pipe('coreferee')
        if ner_nlp:
            self.ner_nlp = spacy.load(ner_nlp)
    
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
        if self.ner:
            return self.ner_doc.ents
        
    def attribute(self, t):
        self.doc = self.nlp(t)
        self.quotes = [q for q in direct_quotations(self.doc)]
        self.persons = [e for e in self.doc.ents if e.label_=='PERSON']
        if self.ner:
            self.ner_doc = self.ner_nlp(t)
            
    def chain_from_token(self, i: Union[int, Token]):
        if isinstance(i, Token):
            i = i.i
        try:
            return next(
                c.index 
                for c in self.doc._.coref_chains.chains 
                if i in [c_.root_index for c_ in c.mentions]
            )
        except StopIteration:
            return
        
    def chain_from_ent(self, ent: Span):
        return [
            chain_n for t in ent 
            if (chain_n:= self.chain_from_token(t)) is not None
        ]
    
    def persons_from_ent(self, ent):
        return [person for person in self.persons if person.text in ent.text]

    def chain_from_person(self, person):
        for p in person:
            resolved = self.doc._.coref_chains.resolve(p)
            if not resolved:
                continue
            else:
                chain_n = self.chain_from_token(resolved[0])
                if not chain_n:
                    continue
                else:
                    return chain_n

    def see_other_people(self, person):
        return [p_ for p_ in self.persons if person.text in p_.text]
    
    def get_ent_chains(self):
        """
        This is stupid, rewrite it.
        """
        self.ent_chains = []
        for e in self.ents:
            chain = self.chain_from_ent(e)
            if chain:
                if chain not in self.ent_chains:
                    self.ent_chains.append(chain[0])

            ppl = self.persons_from_ent(e)
            for person in ppl:
                chain = self.chain_from_person(person)
                if chain and chain not in self.ent_chains:
                    self.ent_chains.append(chain)
                ppl2 = self.see_other_people(person)
                for p in ppl2:
                    chain = self.chain_from_person(person)
                    if chain and chain not in self.ent_chains:
                        self.ent_chains.append(chain)
            
    def speaker_in_ent(self, q):
        for s in q.speaker:
            for e in self.ents:
                if e.start <= s.i <= e.end:
                    return True
        return False
