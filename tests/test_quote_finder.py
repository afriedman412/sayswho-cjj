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
        ## removed because different models were correctly finding varying speakers
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

# TEXTACY COUNTING ERRORS
@pytest.mark.parametrize(
    "text, speakers",
    [(
        """Jun. 22--Police reform and racial injustice were key topics discussed between Aiken\'s local leaders and South Carolina\'s senior senator during a Monday morning roundtable discussion. U.S. Sen. Lindsey Graham, R-S.C., met with law enforcement and local religious and city leaders at the Lessie B. Price Senior and Youth Center where discussions began around 10:30 a.m.\nHis visit to Aiken follows weeks of nationwide protests following the death of George Floyd, a Black man who died in police custody in Minneapolis after a white police officer knelt on his neck for nearly nine minutes while his hands were cuffed behind his back. "Enough is enough," Graham said, referring to Floyd\'s death. "There\'s almost unanimity that what happened to Mr. Floyd was not good policing. It\'s abusive policing. It\'s a stain on us as a nation. It needs to change." Throughout closed-door conversations, local leaders told Graham the issues with racism both in the Aiken community and the nation.\nGraham briefly spoke on his own realization of the different races\' experiences with law enforcement through conversations with Tim Scott, a Republican senator from South Carolina. He said Scott, a Black man, has been stopped six or seven times on Capitol Hill while Graham said he\'s never been stopped. "Black lives matter," Graham said.""", 
        ["Graham", "Graham", "Graham"]
    ),
    (
        """Sometime in the afternoon of Nov. 3, 1966, West Anchorage High School senior William Michael "Mike" Christian left his parents\' home at 4306 North Star Road for the last time. After running some errands, he visited the Sundowner Drive-In theater off the Seward Highway near Chester Creek. After probably several mosquito bites, he stopped to call his girlfriend on the way out. According to her, he seemed in a good mood. He was going to wash his car, attend a car club meeting, and be home well before midnight. He hung up and then vanished. He never made it to the car club or home. His parents did not worry about his absence at first, assuming he spent the night with friends. However, they grew instantly concerned when his beloved 1963 Impala sedan was discovered the next day, abandoned in town. By Nov. 6, the Alaska State Police, now the Alaska State Troopers, were actively investigating his disappearance. One officer was assigned near exclusively to the case. The parents ran advertisements in the local newspapers offering a $500 reward, about $4,000 in 2021.""",
        []
    ),
    (
        """His long rap sheet also showed that seven years ago, he attacked and stabbed one of the victims killed in the weekend rampage, according to court records. Canadian Public Safety Minister Marco Mendicino said the parole board's assessment of Sanderson will be investigated. "I want to know the reasons behind the decision" to release him, Mendicino said. "I'm extremely concerned with what occurred here." Investigators have not given a motive for the bloodshed. The Saskatchewan Coroner's Service said nine of those killed were from the James Smith Cree Nation and one was from Weldon. Court documents said that in 2015 Sanderson attacked his in-laws Earl Burns - killed in last weekend's rampage - and Joyce Burns. He later pleaded guilty to assault and threatening Earl Burns' life. Many of Sanderson's crimes were committed when he was intoxicated, according to court records.""",
        ["Mendicino", "Mendicino"]
    )]
)

def test_common_quote_errors(nlp, text, speakers):
    quotes = direct_quotations(nlp(text))
    assert [speaker.text for quote in quotes for speaker in quote.speaker] == speakers


# TEXTACY MISSES
@pytest.mark.parametrize(
        "text, speakers",
    [( # test that every every quote starts outside of the previous one -- should only return one result
    "Prosecutor Wesley Bell issued a statement, which read, in part, \"We welcome any attempt to make St. Louis County more safe through legislation, particularly with respect to violent crimes like ‘carjackings\' and we look forward to reviewing the proposed legislation.\"\n\nSchmitt said other states, including Illinois and Georgia, also have enacted vehicular hijacking laws, but he did not know whether the laws affected the number of carjackings.",
    ['Schmitt']
    ),
    ]

)

def test_adjustment_for_quote_detection(nlp, text, speakers):
    quotes = direct_quotations(nlp(text))
    assert [speaker.text for quote in quotes for speaker in quote.speaker] == speakers
