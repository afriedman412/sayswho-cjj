"""
Separating the code to load articles into its own doc.

It was getting redundant.
"""
import os
import json
from bs4 import BeautifulSoup
from collections import Counter
import regex as re
import warnings
from .quote_helpers import prep_text_for_quote_detection
from .constants import json_path, file_key

def load_doc(doc_id: str) -> dict:
    """
    Loads the document of doc_id

    Input:
        doc_id (str) - lexis document ID in the format (\S{4}-)-0{4}-00
        df (DataFrame) - filtered dataframe of doc_id the json file (file_name) that contains the document

    Output:
        data (dict) - lexis document query result
    """
    file_name = next(k['file_name'] for k in file_key if k['doc_id']==doc_id)

    data = json.load(open(os.path.join(json_path, file_name)))
    data = next(d for d in data if doc_id in d['ResultId'])
    return data

def full_parse(data: dict, char: str="\n", exp: bool=False) -> str:
    """
    TODO: biggest bodytext or both bodytext?

    - extracts soup from articledata
    - finds biggest bodytext tag in soup
    - makes list of all paragraph text in bodytext
    - joins paragraphs with char and returns

    Input: 
        data (dict) - data of article as extracted from json archive file
        char (str) - character to connect text from all article paragraphs
    Output: 
        full_text - article text, joined by char
    """
    soup = extract_soup(data)
    bodytext = biggest_bodytext(soup)
    full_text = char.join([p.text.strip() for p in bodytext.find_all("p")])
    full_text = prep_text_for_quote_detection(full_text, "\n", exp=exp)
    return full_text

def extract_soup(data: dict):
    """
    Input:
        data (dict) - article data from query json
    
    Output:
        soup (soup) - BeautifulSoup object containing document content

    TODO: sort out lxml roulette
    """
    soup = BeautifulSoup(data['Document']['Content'], features="lxml")
    return soup

def biggest_bodytext(soup):
    """
    Returns longest bodytext object if there is more than one.

    Input:
        soup - soup of article

    Output:
        longest bodytext object in soup
    """
    return sorted(soup.find_all("bodytext"), key= lambda t: len(t.text), reverse=True)[0]

def clean_article(t :str) -> str:
    """
    TODO: Amend with more text to remove

    Input:
        t (str) - article text

    Output:
        t (str) - article with selectd text removed
    """
    t = re.split(r"___ \(c\)20\d{2}", t)[0]
    return t

def get_metadata(soup: BeautifulSoup) -> dict:
    """
    Input:
        soup (soup) - soup for article

    Output:
        metadata (dict) - article info to format article_template.html 
    """
    metadata = {  # your variables to pass to template
        'doc_id': soup.id.text.replace("urn:contentItem:", ""),
        'headline': " - ".join([t.text for t in soup.find("nitf:hedline").find_all()]) if soup.find("nitf:hedline") else "",
        'publication': soup.find("publicationname").text if soup.find("publicationname") else "", 
        'date': soup.datetext.text if soup.datetext else "",
        'byline': soup.nametext.text if soup.nametext else "",
        'wordcount': soup.wordcount['number']
    }
    return metadata 

def quotes_from_soup(html):
    return BeautifulSoup(html).bodytext.find_all("span", attrs={"id":"quote-highlight"})

def audit_rendered_quotes(quotes, rendered_w_quotes):
    """
    For testing!
    """
    for q, r in zip(quotes, quotes_from_soup(rendered_w_quotes)):
        q_ = q.content.text.replace("@", "'")
        r_ = r.text
        if q_!=r_:
            return False
    return True

def audit_parsed_html(file_name, verbose=False):
    soup = BeautifulSoup(open(file_name))
    d = sorted([s for s in soup.find_all('span') if s['id'].startswith("quote-data")], key=lambda s: s['id'])
    h = sorted([s for s in soup.find_all('span') if s['id'].startswith("quote-highlight")], key=lambda s: s['id'])
    if not len(d) == len(h):
        warnings.warn(f"n data ({len(d)}) is not the same as n highlights ({len(h)})")

    for n, (d_, h_) in enumerate(zip(d, h)):
        if d_.text != h_.text:
            print(n)
            if verbose:
                print(d_.text)
                print(h_.text)
                print("---")
    print('done')
    
def o(i):
    os.system(f"open {i}.html")

def pair_quote_matches(quotes, matches):
    """
    Aligns quotes with corresponding matches.
    TODO: move to rendering helpers?
    
    This creates one index for each quote/match pair for Jinja to parse...
    instead of having to deal with index changes when quotes have multiple matches.
    """
    quote_matches = []
    for q in quotes:
        qm = []
        for m in matches:
            if m.content == q.content.text:
                qm.append(m)
        quote_matches.append((q, qm))
    return quote_matches






