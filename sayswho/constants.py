from spacy.symbols import (aux, auxpass, csubj, dobj, neg, nsubj)
import json
import os

# TODO: adjust this for remote functionality
json_path = "../CJJ/query_work_files/query_results_2_2_23/"
file_key = json.load(open('./sayswho/doc_file_key.json'))

"""
Constants for textacy quote identification
"""
_ACTIVE_SUBJ_DEPS = {csubj, nsubj, dobj}
_VERB_MODIFIER_DEPS = {aux, auxpass, neg}

"""
Ordinal points of the token.is_quote characters, matched up by start and end.

source:
switch = "\"\'"
start = "“‘```“‘«‹「『„‚"
end = "”’’’’”’»›」』”’"

"""
QUOTATION_MARK_PAIRS = {
    (34, 34),
    (39, 39),
    (96, 8217),
    (171, 187),
    (8216, 8217),
    (8218, 8217),
    (8220, 8221),
    (8222, 8221),
    (8249, 8250),
    (12300, 12301),
    (12302, 12303),
    (8220, 34),
    (8216, 34),
    (96, 34),
    (8216, 34),
    (171, 34),
    (8249, 34),
    (12300, 34),
    (12302, 34),
    (8222, 34),
    (8218, 34),
    (34, 8221)
    }

_reporting_verbs = {
        "according",
        "accuse",
        "acknowledge",
        "add",
        "admit",
        "agree",
        "allege",
        "announce",
        "argue",
        "ask",
        "assert",
        "believe",
        "blame",
        "charge",
        "cite",
        "claim",
        "complain",
        "concede",
        "conclude",
        "confirm",
        "contend",
        "continue",
        "criticize",
        "declare",
        "decline",
        "deny",
        "describe",
        "disagree",
        "disclose",
        "estimate",
        "explain",
        "fear",
        "hope",
        "insist",
        "maintain",
        "mention",
        "note",
        "observe",
        "order",
        "post",
        "predict",
        "promise",
        "read",
        "recall",
        "recommend",
        "reply",
        "report",
        "say",
        "scream",
        "state",
        "stress",
        "suggest",
        "tell",
        "testify",
        "think",
        "tweet",
        "urge",
        "warn",
        "worry",
        "write",
    }

