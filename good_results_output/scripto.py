from sayswho.sayswho import Attributor, evaluate
from sayswho.article_helpers import load_doc, full_parse, extract_soup, get_metadata
from sayswho.rendering_helpers import render_new
from sayswho.constants import color_key
from tqdm import tqdm
import pandas as pd

df = pd.read_csv("good_articles_subset.csv")

offset=53
a = Attributor()

doc_ids = df['doc_id'].to_list()[9000+offset:]
for doc_id in tqdm(doc_ids):
    # print(doc_id)
    try:
        try:
            data = load_doc(doc_id)
            soup = extract_soup(data)
            metadata = get_metadata(soup)
            t = full_parse(data, "\n")
            a.attribute(t)
            render_new(a, metadata, color_key=color_key, save_file=True)
            with open("good_results_output/sayswho_test_runs.txt", "a+") as f:
                f.write(" | ".join([doc_id, str(evaluate(a))]) + "\n"),
        except Exception as e:
            print(doc_id)
            with open("good_results_output/sayswho_test_run_errors.text", "a+") as f:
                f.write(" | ".join([doc_id, str(e.args)]) + "\n")
    except StopIteration:
        print(doc_id)
        continue