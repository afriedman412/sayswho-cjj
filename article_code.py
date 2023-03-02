"""
Separating the code to load articles into its own doc.

It was getting redundant.
"""
import os
import json
from bs4 import BeautifulSoup
from collections import Counter
import regex as re
from jinja2 import Environment, FileSystemLoader

def full_parse(data, char=" "):
    """
    TODO: biggest bodytext or both bodytext?

    Splits bodytext into paragraphs (by <p> tags) and runs "fix_quote" on paragraph text.

    Input: 
        bodytext (soup.bodytext object)
    Output: 
        texts (list of text)
    """
    texts = []
    soup = extract_soup(data)
    bodytext = biggest_bodytext(soup)
    for p in bodytext.find_all("p"):
        texts.append(fix_quotes(p.text))

    full_text = char.join(texts)
    return clean_article(full_text)

def get_json_data(doc_id: str, engine, path: str="./query_results_2_2_23/" ) -> dict:
    """
    Gets the name of the json file that contains the doc_id.

    Input:
        doc_id (str) - id for document
        engine - sqlalchemy engine
        path (str) - path to directory containing query jsons

    Output:
        dict containing article data
    """
    result = engine.execute(
            f"SELECT `file_name` FROM `articleindex` WHERE `doc_id`='{doc_id}'"
        ).fetchall()
    data = json.load(open(os.path.join(path, result[0][0])))
    return next(d for d in data if doc_id in d['ResultId'])

def extract_soup(data: dict):
    """
    Input:
        data (dict) - article data from query json
    
    Output:
        soup (soup) - BeautifulSoup object containing document content

    TODO: sort out lxml roulette
    """
    soup = BeautifulSoup(data['Document']['Content'], "lxml")
    return soup

def biggest_bodytext(soup):
    """
    Returns longest bodytext object if there is more than one.

    Input:
        soup - soup of article

    Output:
        longbest bodytext object in soup
    """
    return sorted(soup.find_all("bodytext"), key= lambda t: len(t.text), reverse=True)[0]

def fix_quotes(t: str) -> str:
    """
    Adjusts quotation marks that are problematic for spacy and textacy. Designed for maximum throughput not accuracy, so kind of sloppy!

    Input:
        t (str) - article text

    Output:
        t (str) - article text with fixed quotations
    """
    # replace fancy quotes
    t = re.sub(r"[“”]", r'"', t)
    t = re.sub(r"[‘’]", r"'", t)

    # replace (gramatically correct) hanging apostrophes with "'s" so textacy doesnt get fooled
    t = t.replace("s\' ", "s\'s ")

    # add quotation mark to end of any paragraph with an odd number of quotation marks, assuming this is a (gramatically correct) dropped quotation mark
    if Counter(t)['"'] % 2:
        t += '\"'
    for r in re.findall(r"\s(\'.+?\')[\s\"]", t):
        t = t.replace(r, r[1:-1])
    return t

def clean_article(t):
    """
    TODO: Amend with more text to remove

    Input:
        t (str) - article text
    """
    t = re.split(r"___ \(c\)20\d{2}", t)[0]
    return t

def get_metadata(soup):
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

def render_data(data, quotes=None, save_file=False, color='LightGreen'):
    """
    Note that the quotes in here have been fixed -- not exactly equal to article text.
    """
    soup = extract_soup(data)
    metadata = get_metadata(soup)
    metadata['bodytext'] = biggest_bodytext(soup)

    path = './'
    template_name = 'article_template.html'

    rendered = Environment(
        loader=FileSystemLoader(path)
    ).get_template(template_name).render(metadata)
    rendered = fix_quotes(rendered)

    if quotes:
        for q in quotes:
            rendered = rendered.replace(
                q.content.text, 
                f"<span style=\"background-color: {color};\">{q.content.text}</span>"
            )

    file_name = f"{metadata['doc_id']}.html" if save_file else "temp.html"

    with open(file_name, "w+") as f:
        f.write(rendered)







