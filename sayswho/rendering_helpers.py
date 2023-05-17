
from .article_helpers import extract_soup, get_metadata, full_parse
from .sayswho import Attributor
from jinja2 import Environment, FileSystemLoader
from spacy.tokens import Doc, Span, Token
from typing import Iterable
import os

def render_qa(qa: Attributor):
    ents = consolidate_ents(qa)
    return highlight_ents(qa.ner_doc, ents)

def highlight_ents(
        doc: Doc, 
        ents: Iterable[Span]) -> str:
    """
    Creates a HTML-ready text with quotes and LE entities highlighted in CSS.

    Input:
        doc (Doc) - spacy doc
        ents (spans) - list/gen of spans tagged as entities by NER

    Output:
        I can't remember.
    """
    output_text = []
    i = 0

    quote_num = 0

    for e in ents:
        output_text.append(doc[i:e.start].text)
        labeled_text = parse_ent_for_replacing(e, quote_num)
        if e.label_=="QUOTE":
            quote_num += 1
        output_text.append(labeled_text)
        i = e.end
    
    if i < len(doc):
        output_text.append(doc[i:].text)

    return ' '.join(output_text)

def consolidate_ents(qa: Attributor) -> Iterable[Span]:
    """
    Combines and orders quotes and LE ents for HTML parsing.

    Input:
        qa - quoteAttributor object

    Ouput:
        ents - list of QUOTE and LAW ENFORCEMENT entities sorted by start character
    """
    all_ents = list(qa.ner_doc.ents) + [Span(qa.ner_doc, q.content.start, q.content.end, "QUOTE") for q in qa.quotes]
    ents = sorted([e for e in all_ents], key=lambda e: e.start)
    return ents

def parse_ent_for_replacing(e: Span, n: int=0) -> str:
    """
    Formats entity into HTML code to insert via replace during HTML rendering

    Only works for QUOTE and LAW ENFORCEMENT entities

    Input:
        e (Span) - NER-tagged entity
        n (int) - index of entity (for linking)
    
    Output:
        str - HTML-formatted entity or text of span
    """
    if e.label_=="QUOTE":
        return f"<a name=\"quote{n}\"></a><a href=\"#{n}\"><span id=\"QUOTE\" style=\"background-color: lightyellow;\">{e.text}</span></a>"
    
    if e.label_=="LAW ENFORCEMENT":
        return f"<span id=\"{e.label_}\" style=\"color: maroon;\">{e.text}</span>"

    else:
        return e.text


def render_basic(data: dict) -> Iterable:
    """
    Extracts soup and article metatdata from raw article data, renders soup into HTML

    Input:
        data (dict) - lexis document query result

    Output:
        rendered (html) - render HTML of article
        metadata (dict) - article metadata
    """
    soup = extract_soup(data)
    metadata = get_metadata(soup)
    metadata['bodytext'] = "<p>" + full_parse(data, "</p><p>") 
    
    rendered = Environment(
        loader=FileSystemLoader("./")
    ).get_template('article_template.html').render(metadata)
    # rendered = normalize.quotation_marks(rendered)
    return rendered, metadata

def render_quotes(rendered, quotes, color="LightGreen"):
    """
    Renders quotes into HTML.
    TODO: type hints

    Input:
        rendered - HTML-rendered article
        quotes (list) - list of textacy-extract quote triples
        color (str) - css-friendly string for quote highlight color

    Output:
        rendered - HTML-rendered article with highlighted quotes
    """
    if quotes:
        for n, quote in enumerate(quotes):
            q_text = quote.content.text
            rendered = rendered.replace(
                q_text, 
                f"<a name=\"quote{n}\"></a><a href=\"#{n}\"><span id=\"quote-highlight-{n}\" style=\"background-color: {color};\">{q_text}</span></a>",
                1
                )
    return rendered

def render_data(
        data: dict, 
        quotes: Iterable=None, 
        color: str='LightGreen', 
        save_file: bool=False) -> dict:
    """
    Renders article data into HTML and highlights quotes in HTML, returns metadata for further use.
    TODO: integrate this into quoteAttribution

    Input:
        data (dict) - lexis document query result
        quotes (list) - list of textacy-extract quote triples
        color (str) - css-friendly string for quote highlight color
        save_file (bool) - if True, rename saved file from "temp.html" to (doc_id).html

    Output:
        metadata (dict) - article metadata
    """
    rendered, metadata = render_basic(data)
    rendered_w_quotes = render_quotes(rendered, quotes, color)

    file_name = f"{metadata['doc_id']}.html" if save_file else "temp.html"

    with open(file_name, "w+") as f:
        f.write(rendered_w_quotes.replace("@", "'"))

    return metadata