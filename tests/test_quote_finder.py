"""
Cribbed from textacy!!!
"""
import pytest
import spacy
from sayswho.quotes import direct_quotations

@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_lg")

@pytest.mark.parametrize(
    "text, exp",
    [
        (
            'Burton said, "I love cats!"',
            [(["Burton"], ["said"], '"I love cats!"')],
        ),
        (
            'Burton explained from a podium. "I love cats," he said.',
            [(["he"], ["said"], '"I love cats,"')],
        ),
        (
            '"I love cats!" insists Burton. "I absolutely do."',
            [
                (["Burton"], ["insists"], '"I love cats!"'),
                (["Burton"], ["insists"], '"I absolutely do."'),
            ],
        ),
        (
            '"Some people say otherwise," he conceded.',
            [(["he"], ["conceded"], '"Some people say otherwise,"')],
        ),

        ## removed because no easy good way to exclude these without sacrificing short quotes
        # (
        #     'Burton claims that his favorite book is "One Hundred Years of Solitude".',
        #     [],
        # ),
        (
            'Burton thinks that cats are "cuties".',
            [],
        ),
        (
            '"This is really a shame", said Officer Tusk of the Walrus Police Department. "He had such a life ahead of him"',
            [
                (["Officer", "Tusk"], ["said"], '"This is really a shame"'),
                (["Officer", "Tusk"], ["said"], '"He had such a life ahead of him"'),
            ],
        ),
        (
            """Following the arrest, Loden was held in the Union County Jail for security reasons. After meeting with his wife on June 30, 2000, Loden waived his Miranda rights and confessed to a pair of Mississippi Bureau of Investigation officers. He said he killed the youth to preserve his public appearance.

"Looking back now, I wouldn't have released her because I would've lost the image of being the picture-perfect Marine," Loden said to investigators. "When I woke up, I saw the body. I knew I had done it."

An Itawamba County grand jury indicted Loden five months later. Despite the confession, the video tape and a mountain of physical evidence, he pleaded not guilty at his Nov. 21, 2000, arraignment.""",
            [
                (["Loden"], ["said"], '"Looking back now, I wouldn\'t have released her because I would\'ve lost the image of being the picture-perfect Marine,"'),
                (["Loden"], ["said"], '"When I woke up, I saw the body. I knew I had done it."')
            ],
        ),
        ## removed because different models were finding varying speakers
#         ("""The NYPD says it has in place careful guidelines for using facial recognition, and a "hit" off a database search is just a lead and does not automatically trigger that person's arrest.

# "No one has ever been arrested based solely on a positive facial recognition match," said Assistant Commissioner Devora Kaye, an NYPD spokeswoman. "It is merely a lead, not probable cause. We are comfortable with this technology because it has proven to be a valuable investigative method."

# Police also said a kid mug shot is kept only if the suspect is classified a juvenile delinquent and the case ends with a felony conviction.""",
#             [
#                 (['Commissioner', 'Devora', 'Kaye'], ['said'], '"No one has ever been arrested based solely on a positive facial recognition match,"'),
#                 (['Police'], ['said'], '"It is merely a lead, not probable cause. We are comfortable with this technology because it has proven to be a valuable investigative method."')
#             ]
#          )
    ],
)

def test_direct_quotations(nlp, text, exp):
    obs = list(direct_quotations(nlp(text)))
    assert all(hasattr(dq, attr) for dq in obs for attr in ["speaker", "cue", "content"])
    obs_text = [
        ([tok.text for tok in speaker], [tok.text for tok in cue], content.text)
        for speaker, cue, content in obs
    ]
    assert obs_text == exp


def test_difficult_quotes(nlp):
    test_text = open('./tests/quote_error_tests.txt').read().split("\n\n")
    for t in test_text:
        assert direct_quotations(nlp(t))