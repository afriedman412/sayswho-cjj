
from article_code import extract_soup, get_metadata, full_parse
from jinja2 import Environment, FileSystemLoader
from textacy import preprocessing
from spacy.tokens import Span
import os

def render_qa(qa):
    ents = consolidate_ents(qa)
    return highlight_ents(qa.ner_doc, ents)

def highlight_ents(doc, ents):
    """
    Creates a HTML-ready text with quotes and LE entities highlighted in CSS.
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

def consolidate_ents(qa):
    """
    Combines and orders quotes and LE ents for HTML parsing.
    """
    all_ents = list(qa.ner_doc.ents) + [Span(qa.ner_doc, q.content.start, q.content.end, "QUOTE") for q in qa.quotes]
    ents = sorted([e for e in all_ents], key=lambda e: e.start)
    return ents

def parse_ent_for_replacing(e, n=0):
    if e.label_=="QUOTE":
        return f"<a name=\"quote{n}\"></a><a href=\"#{n}\"><span id=\"QUOTE\" style=\"background-color: lightyellow;\">{e.text}</span></a>"
    
    if e.label_=="LAW ENFORCEMENT":
        return f"<span id=\"{e.label_}\" style=\"color: maroon;\">{e.text}</span>"

    else:
        return e.text


def render_basic(data):
    soup = extract_soup(data)
    metadata = get_metadata(soup)
    metadata['bodytext'] = "<p>" + full_parse(data, "</p><p>") 
    
    rendered = Environment(
        loader=FileSystemLoader("./")
    ).get_template('article_template.html').render(metadata)
    rendered = preprocessing.normalize.quotation_marks(rendered)
    return rendered, metadata

def render_quotes(rendered, quote_matches, color="LightGreen"):
    for n, (quote, matches) in enumerate(quote_matches):
        q_text = quote.content.text
        rendered = rendered.replace(
            q_text, 
                f"<a name=\"quote{n}\"></a><a href=\"#{n}\"><span id=\"quote-highlight-{n}\" style=\"background-color: {color};\">{q_text}</span></a>",
                1
            )
    return rendered

def render_data(data, quote_matches=None, color='LightGreen', save_file=False):
    """
    This is true to imported data. Possibly some problems with paragraph breaks, but "fix_quotes" is not run on the incoming text!
    """
    rendered, metadata = render_basic(data)
    rendered_w_quotes = render_quotes(rendered, quote_matches, color)

    file_name = f"{metadata['doc_id']}.html" if save_file else "temp.html"

    with open(file_name, "w+") as f:
        f.write(rendered_w_quotes.replace("@", "'"))

    return metadata

def render_map(metadata, name=False, open_file=False):
    """
    Probably redundant with render_data. Clean up later.
    """
    print('rendering:', metadata['doc_id'])

    rendered = Environment(
        loader=FileSystemLoader('./')
    ).get_template('article_template.html').render(metadata)

    rendered_w_quotes = render_quotes(rendered, metadata['quote_matches'])
    
    file_name = f"{metadata['doc_id'] if name else 'temp'}.html"
    with open(file_name, "w+") as f:
            f.write(rendered_w_quotes)
    
    if open_file:
        os.system(f"open {file_name}")