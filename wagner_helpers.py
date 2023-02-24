from sqlalchemy import create_engine
from jinja2 import Environment, FileSystemLoader
from stanza.server import CoreNLPClient
from bs4 import BeautifulSoup
import regex as re
import json
import os
import requests
import pandas as pd
from typing import Union, List
import textacy
from textacy import extract
from itertools import combinations
from collections import Counter
from tqdm import tqdm

creds = json.load(open("creds.json"))

def get_access_token(
    url="https://auth-api.lexisnexis.com/oauth/v2/token", 
    client_id=creds['LN_CLIENT_ID'],
    client_secret=creds['LN_CLIENT_SECRET']
):
    response = requests.post(
        url,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
    )
    return response.json()["access_token"]
    
class docLoader:
    def __init__(self):
        self.engine = create_engine(
            f"mysql://{creds['MYSQL_USER']}:{creds['MYSQL_PW']}@{creds['MYSQL_URL']}/{creds['MYSQL_DB']}"
            )
        return

    def q(self, query):
        return self.engine.execute(query).fetchall()

    def get_indexers(self, doc_id):
        result = self.q(f'select `cl_scheme`, name, score from indexers where `doc_id` ="{doc_id}" order by score desc')
        return result

    def unpack_doc(self, doc_id):
        data = self.load_file(doc_id)
        return self.unpack_data(data)

    def unpack_data(self, data, return_=True):
        soup = extract_soup(data)
        metadata = get_metadata(soup, True)
        body_soup = metadata.pop("body")
        body_text = format_text(soup)
        if return_:
            return data, soup, metadata, body_soup, body_text
        else:
            self.soup = soup
            self.metadata = metadata
            self.body_soup = body_soup
            self.body_text = body_text

    def load_file(self, doc_id):
        result = self.q(f"SELECT `file_name` FROM `articleindex` WHERE `doc_id`='{doc_id}'")
        path = "./query_results_2_2_23/"
        data = json.load(open(os.path.join(path, result[0][0])))
        return next(d for d in data if doc_id in d['ResultId'])

def extract_soup(data):
    soup = BeautifulSoup(data['Document']['Content'], 'lxml')
    return soup

def indexer_parser(soup):
    """
    Why can't this stand alone?
    """
    output = []
    doc_id = soup.id.text.replace("urn:contentItem:", "")
    for cg in soup.find_all('classificationgroup'):
        cg_scheme = cg['classificationscheme']
        for cl in cg.find_all('classification'):
            cl_scheme = cl['classificationscheme']
            for ci in cl.find_all('classificationitem'):
                dicto = {
                    'doc_id': doc_id,
                    'cg_scheme': cg_scheme,
                    'cl_scheme': cl_scheme,
                }
                if ci.get("score"):
                    dicto['score'] = ci.get('score')
                for cic in ci:
                    dicto[cic.name] = cic.text
                output.append(dicto)
    return output

def get_metadata(soup, body=False):
    context = {  # your variables to pass to template
        'doc_id': soup.id.text.replace("urn:contentItem:", ""),
        'headline': soup.find("nitf:hedline").text if soup.find("nitf:hedline") else "",
        'publication': soup.find("publicationname").text if soup.find("publicationname") else "", 
        'date': soup.datetext.text if soup.datetext else "",
        'byline': soup.nametext.text if soup.nametext else "",
        'wordcount': soup.wordcount['number']
    }
    
    if body:
        context['body'] = [p for t in soup.find_all('bodytext') for p in t.find_all('p')]
    return context 

def fix_quote(t):
    """
    Adjusts quotation marks that are problematic (mostly for spacy)
    """
    t = t.replace("s\' ", "s\'s ").replace("”", '\"').replace("’", "\'")
    t = re.sub(r"[“”]", r'"', t)
    t = re.sub(r"[‘’]", r"'", t)
    if Counter(t)['"'] % 2:
        t += '\"'
    for r in re.findall(r"\s(\'.+?\')[\s\"]", t):
        t = t.replace(r, r[1:-1])
    return t

def fix_quotes(bodytext):
    """
    Splits bodytext into paragraphs (by <p> tags) and runs "fix_quote" on paragraph text.

    Input: bodytext (soup.bodytext object)
    Output: texts (list of text)
    """
    texts = []
    for p in bodytext.find_all("p"):
        texts.append(fix_quote(p.text))
    return texts

def format_text(soup, char=" "):
    """
    TODO: Amend with more text to remove
    """
    t = char.join(fix_quotes(soup.bodytext))
    return re.split(r"___ \(c\)20\d{2}", t)[0]

def render_id(doc_id, dl=None):
    if not dl:
        dl = docLoader()
    data = dl.load_file(doc_id)
    soup = extract_soup(data)
    context = get_metadata(soup)
    context['bodytext'] = soup.bodytext

    path = './'
    filename = 'ln_template.html'

    rendered = Environment(
        loader=FileSystemLoader(path)
    ).get_template(filename).render(context)
    return rendered

def add_color(rendered, quotes, color="lightgreen"):
    for q in quotes:
        rendered = rendered.replace(
            q, f"<span style=\"background-color: {color};\">{q}</span>"
            )
    return rendered

# def save_and_launch(doc_id, rendered):
#     file_name = f"{doc_id}.html"

#     with open(file_name, "w+") as f:
#         f.write(rendered)

#     os.system(f"open {file_name}")

def quick_render(doc_id, dl=None, quotes=None, color='LightGreen'):
    if not dl:
        dl = docLoader()
    data = dl.load_file(doc_id)
    soup = extract_soup(data)
    context = get_metadata(soup)
    context['bodytext'] = soup.bodytext

    path = './'
    filename = 'ln_template.html'

    rendered = Environment(
        loader=FileSystemLoader(path)
    ).get_template(filename).render(context)

    if quotes:
        for q in quotes:
            rendered = rendered.replace(
                q, f"<span style=\"background-color: {color};\">{q}</span>"
            )

    file_name = f"{doc_id}.html"

    with open(file_name, "w+") as f:
        f.write(rendered)

    os.system(f"open {file_name}")

def ext_c_quotes(c_quotes, offset=0):
    """
    Utility function for extracting CoreNLP quotes

    Can prob delete!
    """
    return [(n+offset, q[0], q[2]) for n, q in enumerate(c_quotes[offset:])]



class dbPrep:
    """
    Formatting prep for uploading to DB.
    """
    def __init__(self):
        return

    def classification_group_parser(self, soup):
        output = []
        doc_id = soup.id.text.replace("urn:contentItem:", "")
        for cg in soup.find_all('classificationgroup'):
            cg_scheme = cg['classificationscheme']
            for cl in cg.find_all('classification'):
                cl_scheme = cl['classificationscheme']
                for ci in cl.find_all('classificationitem'):
                    dicto = {
                        'doc_id': doc_id,
                        'cg_scheme': cg_scheme,
                        'cl_scheme': cl_scheme,
                    }
                    if ci.get("score"):
                        dicto['score'] = ci.get('score')
                    for cic in ci:
                        dicto[cic.name] = cic.text
                    output.append(dicto)
        return output

    def extract_metadata(self, soup):
        """
        "get_metadata" above does the same thing
        """
        context = {  # your variables to pass to template
            'doc_id': soup.id.text.replace("urn:contentItem:", ""),
            'headline': soup.find("nitf:hedline").text if soup.find("nitf:hedline") else "",
            'publication': soup.find("publicationname").text,
            'date': soup.datetext.text,
            'byline': soup.nametext.text if soup.nametext else "",
            'wordcount': soup.wordcount['number']
        }
        return context

    def parse_article(self, article):
        soup = self.extract_soup(article)
        metadata = self.extract_metadata(soup)
        indexers = self.classification_group_parser(soup)
        return metadata, indexers

    def parse_json_for_db(self, file_path, return_df=True):
        metadata_ = []
        indexers_ = []
        data = json.load(open(file_path))
        for n, article in enumerate(data):
            try:
                metadata,  indexers = self.parse_article(article)
                metadata_.append(metadata)
                indexers_ += indexers
            except Exception as e:
                print(file_path, n, e.args)
                continue
        if return_df:
            return pd.DataFrame(metadata_), pd.DataFrame(indexers_).rename(
                columns={
                    "classname":"name",
                    "classificationitem":"classification-item"
                }
            )
        else:
            return metadata_, indexers_

"""
Quote attribution code.
"""
def annotate_text(t: Union[List, str], be_quiet: bool=True):
    """
    Annotates either a single message or a list of messages using CoreNLP.

    "be_quiet" turns off verbose mode and enables TQDM.

    TODO: Add variable annotators list.
    """
    if isinstance(t, str):
        t = [t]
    with CoreNLPClient(
            annotators=[
                'tokenize',
                'ssplit',
                'pos',
                'lemma',
                'ner', 
                'parse', 
                'depparse',
                'coref', 
                'quote', 
                'quote.attribution'],
            timeout=300000,
            memory='6G',
            be_quiet=be_quiet
            ) as client:

        if be_quiet:
            annotations = [client.annotate(t_) for t_ in tqdm(t)]
        else:
            annotations = [client.annotate(t_)for t_ in t]
    return annotations

def c_quotes_and_mentions(ann):
    """
    Extracts quotes and mentions from a CoreNLP annotation.
    """
    quotes = [q for q in ann.quote]
    mentions = [m for m in ann.mentions]
    return quotes, mentions

def compile_canonical_entities(mentions):
    """
    To help identify entities that CoreNLP sees as equivalent, for help in parsing roles.

    Inputs:
        mentions: list of CoreNLP mentions

    Outputs:
        canonical_entities: dict with keys = canonicalEntityMentionIndex and value = list of mentions wit that canonicalEntityMentionIndex.
    """
    canonical_entities = {}

    for m in mentions: 
        if m.canonicalEntityMentionIndex in canonical_entities:
            canonical_entities[m.canonicalEntityMentionIndex].append(m)
        else:
            canonical_entities[m.canonicalEntityMentionIndex] = [m]
    return canonical_entities

def process_c_quote(
    quote,
    attrs=[
        'text', 'mention', 'mentionType', 'mentionSieve', 'speaker', 
        'speakerSieve', 'canonicalMention', 'canonicalMentionBegin',
        'tokenBegin', 'tokenEnd'
    ]):
    """
    Converts a CoreNLP quote into a tuple of its attributes.

    Input:
        quote: CoreNLP quote
        attrs: list of strings that are CoreNLP quote attributes

    Output:
        tuple of attributes
    """
    output = [getattr(quote, a) for a in attrs]
    output['tokenLen'] = output['tokenEnd'] - output['tokenBegin'] - 1
    return tuple(output)

def make_c_quotes_df(quotes) -> pd.DataFrame:
    """
    Turns a list of CoreNLP quotes into a dataframe of those quotes.
    """
    return pd.DataFrame(
        [process_c_quote(q) for q in quotes],
        columns=[
            'text', 'mention', 'mentionType', 'mentionSieve', 'speaker', 
            'speakerSieve', 'canonicalMention', 'canonicalMentionBegin',
            'tokenBegin', 'tokenEnd', 'tokenLen'
        ]
    )

def process_t_quote(t_quote):
    """
    Extracts and formats data from a Textacy quote.
    """
    speaker = " ".join([t.text for t in t_quote.speaker])
    cue = " ".join([t.text for t in t_quote.cue])
    context = t_quote.content.text
    
    return (
        " ".join([t.text for t in t_quote.speaker]),
        " ".join([t.text for t in t_quote.cue]),
        t_quote.content.text,
        t_quote.content.start,
        t_quote.content.end
    )

def make_t_quotes_df(t_quotes) -> pd.DataFrame:
    """
    Makes a dataframe out of a list of Textacy quotes.
    """
    t_quote_df = pd.DataFrame(
        [process_t_quote(tq) for tq in t_quotes],
        columns=[
            'speaker', 'cue', 'text', 'tokenBegin', 'tokenEnd'
        ]
    )
    return t_quote_df

def combine_quote_dfs(c_quote_df, t_quote_df) -> pd.DataFrame:
    return c_quote_df[
        ['text', 'speaker', 'speakerSieve', 'tokenBegin', 'tokenEnd', 'canonicalMention',
        'canonicalMentionBegin']
    ].set_index('text').join(
        t_quote_df.set_index('text'),
        lsuffix="_c", rsuffix="_t", how="outer" 
    ).reset_index()

def process_annotations(annotation_data):
    """
    Extracts quotes from annotation data (tuple of doc_id, CoreNLP annotation and Textacy Doc), returns aligned dataframe.

    TODO: Make exceptions for quote counting errors
    TODO: Count articles w no quotes and verify with unqiue ids in df
    """
    quote_dfs = []
    dl = docLoader()
    for a in annotation_data:
        i = a[0]
        quotes, mentions = c_quotes_and_mentions(a[1])
        c_quote_df = make_c_quotes_df(quotes)
        t_quotes = [q for q in extract.triples.direct_quotations(a[2])]
        t_quote_df = make_t_quotes_df(t_quotes)
        print("c quotes:", len(c_quote_df), "/ t quotes:", len(t_quote_df))
        quote_df = combine_quote_dfs(c_quote_df, t_quote_df)
        quote_df['doc_id'] = i
        quote_dfs.append(quote_df)

        r = render_id(i, dl)
        r = fix_quote(r)
        r = add_color(r, quote_df['text'])
        file_name = f"{i}.html"
        with open(file_name, "w+") as f:
            f.write(r)

    
    big_quote_df = pd.concat(quote_dfs)
    big_quote_df['token_diff'] = big_quote_df.apply(
        lambda r: (r['tokenBegin_t'] + r['tokenEnd_t']) - (r['tokenBegin_c'] + r['tokenEnd_c']), 1
    )
    big_quote_df['speaker_match'] = big_quote_df.apply(
        lambda r: r['speaker_c'] == r['speaker_t'], 1
    )
    return big_quote_df

def title_to_name(mentions):
    """
    limited utility
    """
    for m in mentions:
        if m.ner == 'TITLE':
            print(
                m.entityMentionIndex+1, ' '.join(
                    [m.entityMentionText, mentions[m.entityMentionIndex+1].entityMentionText]
                    )
                )


            

"""
Textacy quote attribution helpers

Should prob split these into diff docs or classes!
"""

class textacyHelper:
    """
    Specifically set up to test different models against each other -- redo this for more utility in the future.
    """
    def __init__(self):
        self.ks = ['sm', 'md', 'lg']
        self.en = {k: textacy.load_spacy_lang(f"en_core_web_{k}") for k in self.ks}


    def compare_quotes(self, quotes, l):
        all_equal=True
        for q in range(l):
            for c in combinations(self.ks, 2):
                for n in range(3):
                    if not str(quotes[c[0]][q][n]) == str(quotes[c[1]][q][n]):
                        all_equal=False
                        print(c[0], c[1], f'quote: {q}', n)
        return all_equal
                        
    def len_check(self, quotes):
        l = dict(zip(self.ks, [len(quotes[k]) for k in self.ks]))
        if len(set(l.values())) < 2:
            return list(l.values())[0]
        else:
            return l

    def make_doc(self, doc_id):
        dl = docLoader()
        doc_data = dl.unpack_doc(doc_id)
        doc = {k: textacy.make_spacy_doc(doc_data[-1], lang=self.en[k]) for k in self.ks}
        return doc

    def make_quotes(self, doc):
        quotes = {k: [q for q in extract.triples.direct_quotations(doc[k])] for k in self.ks}
        return quotes

    def full_compare(self, doc_id):
        try:
            self.doc = self.make_doc(doc_id)
            self.quotes = self.make_quotes(self.doc)
            l = self.len_check(self.quotes)
            if isinstance(l, dict):
                return " ".join([doc_id, 'diff', l])
            else:
                all_equal = self.compare_quotes(self.quotes, l)
                if all_equal:
                    return ' '.join([doc_id, "all equal", l, "quotes", len(self.doc['sm']), "tokens"])
                else:
                    return ' '.join([doc_id, "unequal", l, "quotes", len(self.doc['sm']), "tokens"])
        except ValueError as e:
            return ' '.join([doc_id, "quote error", e.args[0].split(", ")[0]])
            
    def show_quotes(self, quotes, offset=0):
        for n, q in enumerate(quotes[offset:]):
            print(f"--{n+offset}--")
            print(q[0])
            print(q[2])
        
    def ext_quotes(self, quotes, offset=0):
        return [(n+offset, q[0], q[2]) for n, q in enumerate(quotes[offset:])]