
from .article_helpers import extract_soup, get_metadata, full_parse
from .sayswho import Attributor, evaluate
from jinja2 import Environment, FileSystemLoader
from spacy.tokens import Doc, Span
from spacy import displacy
from .quote_helpers import DQTriple
from typing import Iterable
import regex as re

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

def render_new(a: Attributor, metadata: dict, color_key: dict, save_file: bool=False):
    metadata['bodytext'] = render_attr_with_highlights(a, color_key)
    metadata['quotes'] = yield_quotes(a)
    metadata['score'] = {k:getattr(evaluate(a),k) for k in ['n_quotes', 'n_ent_quotes', 'n_ents_quoted']}
    rendered = Environment(
        loader=FileSystemLoader("./")
    ).get_template('article_template.html').render(metadata)
    
    file_name = f"{metadata['doc_id']}.html" if save_file else "temp.html"

    with open(file_name, "w+") as f:
        try:
            f.write(rendered)
        except UnicodeDecodeError:
            rendered = re.sub("\u2014", "-", rendered)
            f.write(rendered)
    return

def yield_quotes(a):
    for quote_index, quote in enumerate(a.quotes):
        for m in (qm for qm in a.quote_matches if qm.quote_index==quote_index):
            base_dict = render_quote(quote)
            base_dict['cluster_index'] = m.cluster_index
            base_dict['cluster'] = ', '.join(
                set([c.text for c in a.clusters[m.cluster_index] if c[0].pos_ !="PRON"])
            )
            try:
                ent = next(em for em in a.ent_matches if em.quote_index==m.quote_index)
                base_dict['ent_index'] = ent.ent_index
                base_dict['ent'] = a.ents[ent.ent_index].text
            except StopIteration:
                pass
            yield base_dict

def render_quote_match(a, quote_match):
    quote = render_quote(a.quotes[quote_match.quote_index])
    quote['cluster'] = ', '.join(
        set([c.text for c in a.clusters[quote_match.cluster_index] if c[0].pos_ !="PRON"])
    )
    return quote

def render_quote(quote):
    return {
            "content": quote.content,
            "cue": "".join([t.text_with_ws for t in quote.cue]),
    }

def get_ent_quote_indexes(a: Attributor) -> list:
    ent_idxs = [((e.start, e.end), e.label_, n) for n, e in enumerate(a.ents)]
    quote_idxs = [((q.content.start, q.content.end), "QUOTE", n) for n, q in enumerate(a.quotes)]
    indexes = sorted(ent_idxs+quote_idxs, key=lambda i: i[0])
    return indexes

def render_attr_with_highlights(a: Attributor, color_key: dict) -> str:
    indexes = get_ent_quote_indexes(a)
    text_bucket = ["<p>"]
    for token in a.doc:
        coded = False
        if token.text == "\n":
            token_text = "</p><p>"
        else:
            token_text = token.text_with_ws
        for index_match in (i_ for i_ in indexes if token.i in i_[0]):
            coded = True
            match_indexes, label, n = index_match
            start = not match_indexes.index(token.i)
            html_code = generate_code(n, label, start, color_key)
            text_bucket.append(html_code + token_text)
        if not coded:
            text_bucket.append(token_text)

    text_bucket.append("</p>")
    return "".join(text_bucket)

def generate_code(
    n: int, label: str, start: bool=True, color_key: dict={}
) -> str:
    color = color_key.get(label)
    if label.lower()=='quote':
        if start:
            return f"<a name=\"quote{n}\"></a><a href=\"#{n}\"><span id=\"QUOTE\" style=\"background-color: {color};\">"
        else:
            return "</span></a>"
    else:
        if start:
            return f"<span id=\"{label}\" style=\"color: {color};\">"
        else:
            return "</span>"
        
def double_viz(a: Attributor):
    """
    Displacy visualization of all quotes and law enforcement entities.

    TODO: Doesn't recognize line breaks, and that is a problem.
    """
    a.doc.spans['custom'] = [
        Span(a.doc, e.start, e.end, "LAW ENFORCEMENT") for e in a.ner_doc.ents
    ] + [
        Span(a.doc, q.content.start, q.content.end, 'QUOTE') for q in a.quotes
    ]
    
    html = displacy.render(
        a.doc, 
        style="span", 
        options={
            "spans_key":"custom",
            "colors": {
                "LAW ENFORCEMENT": "lightgreen",
                "QUOTE": "lightblue"
                    }
        },
        page=True
    )
    return




