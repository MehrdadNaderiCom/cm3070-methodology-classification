"""Tests for the data layer, taxonomy and evaluation harness.

The parity tests in test_prototype_parity.py guard the published numbers. These
guard the rules that decide which rows a result is computed on and how labels are
allowed to come into existence, which is where a result would go wrong quietly
rather than loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdfm import taxonomy
from cdfm.data import schema
from cdfm.evaluation import folds
from cdfm.paths import DATA_PROCESSED
from cdfm.tasks import MINIMUM_SUPPORT, TASKS, prepare

CORPUS_PATH = DATA_PROCESSED / "corpus.csv"


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    if not CORPUS_PATH.exists():
        pytest.skip("corpus not built")
    return pd.read_csv(CORPUS_PATH, keep_default_na=False)


def frame_with(labels: list[str], **overrides) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(labels):
        row = {column: "" for column in schema.ALL_COLUMNS}
        row.update(
            {
                "paper_id": f"p{index}",
                "source": "test",
                "title": f"Title {index}",
                "abstract": "x" * 400,
                "year": 2024,
                "methodology_label_5": label,
                "methodology_source": schema.PROVENANCE_HUMAN,
            }
        )
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


class TestFolds:
    def test_never_exceeds_the_smallest_class(self):
        labels = ["a"] * 20 + ["b"] * 3
        assert folds.safe_split_count(labels, requested=5) == 3

    def test_uses_the_request_when_classes_allow(self):
        labels = ["a"] * 20 + ["b"] * 20
        assert folds.safe_split_count(labels, requested=5) == 5

    def test_singleton_fails_safely(self):
        labels = ["a"] * 20 + ["b"] * 1
        with pytest.raises(ValueError, match="singleton"):
            folds.safe_split_count(labels, requested=5)

    def test_single_class_is_rejected(self):
        with pytest.raises(ValueError):
            folds.safe_split_count(["a"] * 10)

    def test_folds_are_reproducible(self):
        labels = np.array(["a"] * 20 + ["b"] * 20)
        first = list(folds.make_folds(labels).split(np.zeros(len(labels)), labels))
        second = list(folds.make_folds(labels).split(np.zeros(len(labels)), labels))
        for (train_a, test_a), (train_b, test_b) in zip(first, second):
            assert np.array_equal(train_a, train_b)
            assert np.array_equal(test_a, test_b)


class TestTaskPreparation:
    def test_thin_classes_are_dropped_when_enough_survive(self):
        frame = frame_with(["a"] * 20 + ["b"] * 20 + ["c"] * 3)
        prepared = prepare(frame, TASKS["methodology5"])
        assert prepared.dropped_classes == {"c": 3}
        assert set(prepared.labels.unique()) == {"a", "b"}
        assert "excluded" in prepared.note()

    def test_nothing_is_dropped_when_too_few_would_survive(self):
        """The prototype's own task has no class above twelve examples."""
        frame = frame_with(["a"] * 12 + ["b"] * 12 + ["c"] * 8)
        prepared = prepare(frame, TASKS["methodology5"])
        assert prepared.dropped_classes == {}
        assert len(prepared.frame) == 32
        assert set(prepared.under_supported) == {"a", "b", "c"}
        assert "wide uncertainty" in prepared.note()

    def test_unlabelled_rows_are_excluded(self):
        frame = frame_with(["a"] * 20 + ["b"] * 20 + [""] * 10)
        prepared = prepare(frame, TASKS["methodology5"])
        assert len(prepared.frame) == 40

    def test_text_is_title_then_abstract(self):
        frame = frame_with(["a"] * 20 + ["b"] * 20)
        prepared = prepare(frame, TASKS["methodology5"])
        assert prepared.text.iloc[0].startswith("Title 0. ")

    def test_minimum_support_matches_the_risk_register(self):
        assert MINIMUM_SUPPORT == 15


class TestFiveClassDerivation:
    def test_direct_mappings(self):
        collapse = taxonomy.collapse_map()
        assert schema.derive_five_class("Experimental", "", collapse) == "Experimental"
        assert schema.derive_five_class("Literature Survey", "", collapse) == "Review"
        assert schema.derive_five_class("Systematic Literature Review", "", collapse) == "Review"

    def test_mixed_methods_uses_its_dominant_component(self):
        collapse = taxonomy.collapse_map()
        assert schema.derive_five_class("Mixed Methods", "Case Study", collapse) == "Case Study"
        assert (
            schema.derive_five_class("Mixed Methods", "Literature Survey", collapse) == "Review"
        )

    def test_mixed_methods_without_a_dominant_yields_nothing(self):
        """Better an empty cell than a guessed label."""
        assert schema.derive_five_class("Mixed Methods", "", taxonomy.collapse_map()) == ""

    def test_every_seven_class_label_has_a_target(self):
        collapse = taxonomy.collapse_map()
        for name in taxonomy.methodology_names(7):
            assert name in collapse

    def test_collapse_targets_are_real_five_class_labels(self):
        allowed = set(taxonomy.methodology_names(5)) | {"DOMINANT"}
        assert set(taxonomy.collapse_map().values()) <= allowed


class TestSchemaValidation:
    def test_a_machine_proposal_is_not_a_label(self):
        frame = frame_with(["a"] * 20 + ["b"] * 20)
        frame["methodology_label_7"] = "Experimental"
        frame["methodology_source"] = schema.PROVENANCE_PROPOSED
        report = schema.validate(frame)
        assert not report.ok
        assert any("proposal" in message for message in report.errors)

    def test_a_label_without_a_provenance_is_rejected(self):
        frame = frame_with(["a"] * 20 + ["b"] * 20)
        frame["methodology_source"] = ""
        report = schema.validate(frame)
        assert not report.ok

    def test_duplicate_identifiers_are_rejected(self):
        frame = frame_with(["a"] * 20 + ["b"] * 20)
        frame.loc[1, "paper_id"] = frame.loc[0, "paper_id"]
        assert not schema.validate(frame).ok

    def test_a_clean_frame_passes(self):
        frame = frame_with(["a"] * 20 + ["b"] * 20)
        assert schema.validate(frame).ok


class TestProposals:
    def test_a_systematic_review_is_recognised(self):
        proposal = taxonomy.propose(
            "A Systematic Literature Review of Code Summarisation",
            "We conducted a systematic literature review. Inclusion and exclusion "
            "criteria were applied to primary studies retrieved from Scopus. " + "x" * 200,
        )
        assert proposal.label == "Systematic Literature Review"
        assert proposal.evidence[proposal.label]

    def test_evidence_quotes_the_passage_it_came_from(self):
        proposal = taxonomy.propose(
            "A case study of one deployment",
            "We report on the deployment of a scheduling system at a large company. " + "x" * 200,
        )
        for items in proposal.evidence.values():
            for item in items:
                assert item.sentence
                assert item.matched.lower() in item.sentence.lower()

    def test_no_cues_means_no_proposal(self):
        proposal = taxonomy.propose("Untitled", "Nothing here resembles a cue phrase at all.")
        assert proposal.label == ""
        assert not proposal.is_confident

    def test_proposals_are_deterministic(self):
        args = ("A survey of methods", "This paper surveys existing approaches. " + "x" * 200)
        assert taxonomy.propose(*args).scores == taxonomy.propose(*args).scores

    def test_highlight_spans_never_overlap(self):
        spans = taxonomy.cue_spans(
            "A systematic literature review and case study",
            "We survey the literature and we present a tool. We evaluate it. " + "x" * 200,
        )
        for (_, first_end, _), (second_start, _, _) in zip(spans, spans[1:]):
            assert second_start >= first_end


def test_who_decided_a_label_agrees_with_how_it_was_produced(corpus: pd.DataFrame):
    """`annotator` and `methodology_source` must describe the same fact.

    Decision log entry D10 records who decided and how. The check is that the
    two columns agree: every decided label is `human` and the annotator is
    `author`.
    """
    labelled = corpus[corpus["methodology_source"].astype(str).str.strip().ne("")]
    named = labelled[labelled["annotator"].astype(str).str.strip().ne("")]

    machine = named[named["methodology_source"].isin(schema.MACHINE_DERIVED_PROVENANCES)]
    assert machine.empty, (
        f"{len(machine)} rows carry a machine derived provenance where a decided label "
        "is expected"
    )

    decided = named[named["methodology_source"].isin(schema.HUMAN_VERIFIED_PROVENANCES)]
    who = sorted(set(decided["annotator"].astype(str)))
    assert who == ["author"], (
        "methodology_source says a person decided these labels while annotator names "
        f"{who}. One of the two columns is wrong. See decision log D10."
    )

    sources = sorted(set(decided["methodology_source"].astype(str)))
    assert sources == [schema.PROVENANCE_HUMAN], (
        "every methodology label in this corpus is an author decision recorded as "
        f"{schema.PROVENANCE_HUMAN}. Found {sources}. See decision log D10."
    )


class TestCorpusIntegrity:
    def test_identifiers_are_unique(self, corpus):
        assert not corpus["paper_id"].duplicated().any()

    def test_all_three_disciplines_are_populated(self, corpus):
        counts = corpus["discipline_label"].value_counts()
        for name in taxonomy.discipline_names():
            assert counts.get(name, 0) >= 40, f"{name} has too few papers to evaluate"

    def test_every_methodology_label_is_human_verified(self, corpus):
        """A cue matching suggestion must never stand where a decision belongs.

        `proposed` means the pre-sort script guessed and nobody looked. It is
        never allowed where a label is expected.
        """
        labelled = corpus[corpus["methodology_label_7"].astype(str).str.strip().ne("")]
        sources = set(labelled["methodology_source"].unique())
        assert sources == {schema.PROVENANCE_HUMAN}
        assert not sources & set(schema.MACHINE_DERIVED_PROVENANCES)

    def test_provenance_is_stated_by_validation(self, corpus):
        """Provenance has to be printed, not reconstructed later."""
        report = schema.validate(corpus)
        assert any(
            "annotated and verified by the author" in message for message in report.notes
        ), "validation did not state how the methodology labels were produced"

    def test_field_labels_are_from_the_declared_taxonomy(self, corpus):
        used = {value for value in corpus["field_label"].astype(str) if value.strip()}
        assert used <= set(taxonomy.field_names())

    def test_five_class_labels_agree_with_the_seven_class_collapse(self, corpus):
        collapse = taxonomy.collapse_map()
        both = corpus[
            corpus["methodology_label_7"].astype(str).str.strip().ne("")
            & corpus["methodology_label_5"].astype(str).str.strip().ne("")
        ]
        for _, row in both.iterrows():
            expected = schema.derive_five_class(
                str(row["methodology_label_7"]), str(row["methodology_dominant"]), collapse
            )
            assert str(row["methodology_label_5"]) == expected, row["paper_id"]

    def test_abstracts_are_long_enough_to_carry_evidence(self, corpus):
        assert corpus["abstract"].astype(str).str.len().min() >= 300
