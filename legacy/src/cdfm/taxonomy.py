"""Load the taxonomy and score cue phrases as a reading aid.

The pre-sort exists to make a person's decision faster, not to make it for them.
It reads an abstract, finds the phrases the annotation guide names as evidence
for each class, and reports what it found along with the passage it found it in.
The annotator sees the evidence and decides.

Nothing here writes a methodology label. A proposal carries the provenance
`proposed`, and `cdfm.data.schema.validate` fails any dataset in which a
proposal is still standing where a label should be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from cdfm.paths import CONFIG

TAXONOMY_PATH = CONFIG / "taxonomy.yaml"

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Evidence:
    """One cue phrase found in a paper, with the passage it appeared in."""

    cue: str
    matched: str
    sentence: str
    weight: float


@dataclass
class Proposal:
    """A suggested starting point for an annotator, with its reasoning."""

    label: str
    runner_up: str
    scores: dict[str, float]
    evidence: dict[str, list[Evidence]] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        """Gap between first and second place. A small gap means a hard case."""
        ordered = sorted(self.scores.values(), reverse=True)
        if len(ordered) < 2:
            return 0.0
        return ordered[0] - ordered[1]

    @property
    def is_confident(self) -> bool:
        return self.scores.get(self.label, 0.0) > 0 and self.margin >= 1.0


@dataclass(frozen=True)
class MethodologyClass:
    name: str
    definition: str
    boundary: str
    maps_to_5: str
    positive: tuple[tuple[str, re.Pattern[str], float], ...]
    negative: tuple[tuple[str, re.Pattern[str], float], ...]
    requires_dominant: bool


def _compile(cues: list[str] | None) -> tuple[tuple[str, re.Pattern[str], float], ...]:
    """Compile cue phrases, weighting a longer phrase above a shorter one.

    A three word phrase such as `systematic literature review` is far stronger
    evidence than a single word such as `survey`, so it counts for more.
    """
    compiled = []
    for cue in cues or []:
        weight = float(len(re.sub(r"[^\w\s]", " ", cue).split()))
        compiled.append((cue, re.compile(cue, re.IGNORECASE), max(1.0, weight)))
    return tuple(compiled)


@lru_cache(maxsize=1)
def load(path: str | None = None) -> dict:
    source = Path(path) if path else TAXONOMY_PATH
    with source.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def methodology_classes() -> tuple[MethodologyClass, ...]:
    config = load()["methodology"]
    return tuple(
        MethodologyClass(
            name=entry["name"],
            definition=" ".join(entry["definition"].split()),
            boundary=" ".join(entry.get("boundary", "").split()),
            maps_to_5=entry["maps_to_5"],
            positive=_compile(entry.get("positive_cues")),
            negative=_compile(entry.get("negative_cues")),
            requires_dominant=bool(entry.get("requires_dominant", False)),
        )
        for entry in config["classes"]
    )


def collapse_map() -> dict[str, str]:
    return dict(load()["methodology"]["collapse_to_5"])


def methodology_names(granularity: int = 7) -> list[str]:
    if granularity == 5:
        return list(load()["methodology"]["classes_5"])
    return [entry.name for entry in methodology_classes()]


def discipline_names() -> list[str]:
    return [entry["name"] for entry in load()["discipline"]["classes"]]


def field_names() -> list[str]:
    return list(load()["field"]["classes"])


def _sentence_for(text: str, position: int) -> str:
    """Return the sentence a match sits in, so the annotator reads it in context."""
    start = text.rfind(". ", 0, position)
    start = 0 if start == -1 else start + 2
    end = text.find(". ", position)
    end = len(text) if end == -1 else end + 1
    return " ".join(text[start:end].split())


def propose(title: str, abstract: str) -> Proposal:
    """Score every methodology class against a paper and return the ranking.

    The score is the summed weight of positive cues found, less the summed weight
    of negative cues. It is a reading aid built from the annotation guide, not a
    trained model, and it is deliberately simple enough that a person can see why
    it said what it said.
    """
    text = f"{title}. {abstract}".strip()

    scores: dict[str, float] = {}
    evidence: dict[str, list[Evidence]] = {}

    for entry in methodology_classes():
        total = 0.0
        found: list[Evidence] = []

        for cue, pattern, weight in entry.positive:
            for match in pattern.finditer(text):
                total += weight
                found.append(
                    Evidence(
                        cue=cue,
                        matched=match.group(0),
                        sentence=_sentence_for(text, match.start()),
                        weight=weight,
                    )
                )
                break

        for cue, pattern, weight in entry.negative:
            for match in pattern.finditer(text):
                total -= weight
                found.append(
                    Evidence(
                        cue=cue,
                        matched=match.group(0),
                        sentence=_sentence_for(text, match.start()),
                        weight=-weight,
                    )
                )
                break

        scores[entry.name] = total
        if found:
            evidence[entry.name] = sorted(found, key=lambda item: -abs(item.weight))

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    label = ranked[0][0] if ranked and ranked[0][1] > 0 else ""
    runner_up = ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0 else ""

    return Proposal(label=label, runner_up=runner_up, scores=scores, evidence=evidence)


def cue_spans(title: str, abstract: str) -> list[tuple[int, int, str]]:
    """Character spans of every cue match, for highlighting in the review tool.

    Returns (start, end, class_name) over the combined title and abstract text,
    with overlaps removed so that highlighting cannot double paint a passage.
    """
    text = f"{title}. {abstract}".strip()
    spans: list[tuple[int, int, str]] = []

    for entry in methodology_classes():
        for _, pattern, _ in entry.positive:
            for match in pattern.finditer(text):
                spans.append((match.start(), match.end(), entry.name))
                break

    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    merged: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, name in spans:
        if start >= last_end:
            merged.append((start, end, name))
            last_end = end
    return merged
