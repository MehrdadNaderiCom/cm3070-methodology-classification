"""Transparent cue scoring for five-class methodology classification.

The lexicon encodes the positive-evidence column of the annotation guide.
It is shared by the study-path Cue candidate and by guide-consistency checks,
so the rule set is defined once and cannot drift between components.
"""
from __future__ import annotations

import re

CLASSES = ("Case Study", "Conceptual", "Design/Engineering", "Experimental", "Review")

# Phrase cues with weights. Boundary phrases from the guide are deliberately
# weaker than primary-method phrases.
CUES = {
    "Case Study": [
        (r"\bcase stud(?:y|ies)\b", 3.0), (r"\bin[- ]depth (?:study|interviews?)\b", 2.5),
        (r"\bwe interviewed\b", 3.0), (r"\binterviews? with\b", 2.0),
        (r"\bethnograph(?:ic|y)\b", 2.5), (r"\bfield stud(?:y|ies)\b", 2.0),
        (r"\bsingle case\b", 2.5), (r"\bmulti(?:ple)?[- ]case\b", 2.0),
        (r"\breal[- ]world (?:setting|deployment|context)s?\b", 1.0),
        (r"\borganisation(?:al)? (?:study|context)s?\b", 1.0),
    ],
    "Conceptual": [
        (r"\bwe argue\b", 3.0), (r"\bconceptual (?:framework|model|account)\b", 2.0),
        (r"\bwe propose (?:a|an) (?:conceptual|theoretical)\b", 2.5),
        (r"\btheoretical (?:framework|model|account)\b", 2.0),
        (r"\bposition paper\b", 3.0), (r"\ba taxonomy of\b", 2.0),
        (r"\btowards a\b", 1.5), (r"\bwe (?:introduce|define) the (?:notion|concept)\b", 2.0),
        (r"\bconceptuali[sz](?:e|ing|ation)\b", 1.5),
    ],
    "Design/Engineering": [
        (r"\bwe (?:present|propose|develop|design|build|implement) (?:a|an) (?:system|tool|framework|method|approach|pipeline|library|toolkit|model|architecture|prototype|algorithm)\b", 3.0),
        (r"\bwe (?:present|develop|design|build|implement)\b", 1.5),
        (r"\bour (?:system|tool|approach|method|framework|prototype)\b", 1.5),
        (r"\b(?:software|web|open[- ]source) (?:tool|system|library|framework|toolkit)\b", 2.0),
        (r"\barchitecture\b", 1.0), (r"\bprototype\b", 1.5),
        (r"\b(?:code|implementation) is (?:available|open)\b", 1.0),
    ],
    "Experimental": [
        (r"\bwe (?:conduct|carried out|ran) (?:a|an) (?:empirical|experimental|user|simulation)? ?stud(?:y|ies|ies)?\b", 3.0),
        (r"\buser stud(?:y|ies)\b", 2.5), (r"\bparticipants?\b", 1.5),
        (r"\bexperiment(?:s|al)?\b", 1.5), (r"\bwe (?:evaluate|assess|measure|compare|analyse|analyze)\b", 2.0),
        (r"\bquestionnaire(?:s)?\b", 2.0), (r"\bstatistic(?:al|ally)\b", 1.5),
        (r"\b(?:accuracy|precision|recall|f1|false positive|significant(?:ly)?)\b", 1.0),
        (r"\bp\s*[<=]\s*0?\.\d+\b", 2.0), (r"\bdataset\b", 1.0),
        (r"\bresults? (?:show|indicate|suggest)\b", 1.5),
        (r"\bempirical(?:ly)? (?:study|investigation|evidence|analysis)\b", 2.0),
    ],
    "Review": [
        (r"\bsystematic (?:literature )?review\b", 3.5), (r"\bliterature (?:review|survey)\b", 3.0),
        (r"\bmeta[- ]analysis\b", 3.0), (r"\bscoping review\b", 3.0),
        (r"\bwe (?:review|survey|synthesi[sz]e) (?:the|existing|prior|related)\b", 3.0),
        (r"\b(?:mapping|bibliometric|scientometric) (?:study|review|analysis)\b", 2.5),
        (r"\bstate[- ]of[- ]the[- ]art\b", 1.5), (r"\b(?:we )?review of\b", 1.0),
        (r"\bstudies (?:were|was) (?:selected|identified|included)\b", 2.5),
    ],
}

_COMPILED = {c: [(re.compile(p, re.I), w) for p, w in patterns] for c, patterns in CUES.items()}


def cue_scores(text):
    """Return unnormalised cue scores per class for one title+abstract string."""
    scores = {}
    for cls, patterns in _COMPILED.items():
        total = 0.0
        for pattern, weight in patterns:
            total += weight * len(pattern.findall(text))
        scores[cls] = total
    return scores


def cue_frame_scores(titles, abstracts):
    return [cue_scores(f"{t}. {a}") for t, a in zip(titles, abstracts)]


