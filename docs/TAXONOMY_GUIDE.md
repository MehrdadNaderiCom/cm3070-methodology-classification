# Methodology annotation guide

Version 1.1 | Primary five-class reference | Companion to `PROTOCOL.md`

## Version history

- **1.0 (3 August 2026):** initial locked guide used for the pilot.
- **1.1 (6 August 2026):** after the pilot, the Experimental/Case Study boundary was clarified for outcome-comparison studies of a single organization, and guidance was added on recording the ambiguity flag where the breadth of Experimental conceals a material distinction. No class definitions were renamed, so no symmetric re-review was required.

## Decision unit

Classify the primary research contribution of one paper, not its topic, venue, title keyword or the software used. A computing paper can mention experiments while principally presenting a design, or describe an application while principally developing a conceptual account. Record the activity that establishes the central claim.

Read Methods, Methodology or Study Design first, then data collection, analysis, Results and Evaluation. Read the introduction and conclusion to resolve the contribution, and use the abstract as context rather than as the sole evidence. Record the sections actually accessed. If a full text is unavailable, mark insufficient evidence and follow the exclusion/escalation policy rather than quietly using abstract-only evidence.

## Five-class decision boundaries

| Class | Positive evidence | Closest boundary | Do not infer from |
|---|---|---|---|
| Case Study | A bounded real-world case is the unit of analysis; contextual evidence supports an account of that case. | Experimental when a bounded deployment is used chiefly to estimate or compare an outcome; Design/Engineering when the artifact itself is the contribution. | The word "case" in a title or an illustrative example. |
| Conceptual | An argument, theory, conceptual model or framework is established primarily through reasoning rather than a new empirical evaluation or implemented artifact. | Design/Engineering when a framework is actually instantiated and evaluated; Review when evidence synthesis is the primary method. | The word "framework" alone. |
| Design/Engineering | The central contribution is an artifact, method, system or technical design; evaluation supports its construction or utility. | Experimental when estimating a phenomenon or comparing interventions is primary rather than validating a new artifact. | Presence of code, a prototype or an architecture diagram alone. |
| Experimental | The inherited category broadly covers empirical measurement, comparison or hypothesis evaluation, including some non-randomized, observational and questionnaire studies. | Case Study for bounded contextual inquiry; Design/Engineering for artifact-led work. | A claim of randomized or causal experimentation. The inherited class name does not imply either. |
| Review | The primary contribution is synthesis of existing publications rather than new observations or an implemented artifact. | Conceptual when literature supports a primarily argumentative contribution rather than a synthesis procedure. | A literature-review section present in an otherwise empirical paper. |

The breadth of Experimental is a limitation inherited from the original taxonomy. Preserve the operational meaning for comparability, but flag cases where the name conceals a material distinction such as observational versus intervention research. Do not rename a class halfway through annotation without a versioned mapping and symmetric review.

## Secondary seven-class field

Use the inherited seven-class values: Case Study, Theoretical/Conceptual, Design Science/Engineering, Experimental, Systematic Review, Literature Survey and Mixed Methods. A Systematic Review requires evidence of a repeatable search and selection procedure; Literature Survey covers narrative synthesis without that demonstrated protocol.

Case Study, Theoretical/Conceptual, Design Science/Engineering and Experimental map naturally to their corresponding primary classes. Systematic Review and Literature Survey map to Review. Mixed Methods does not have a universal automatic mapping: record each component and identify the dominant contribution using the primary guide, or preserve an unresolved primary label when the evidence does not support dominance.

Discipline and field are secondary contextual fields. They must not determine methodology, and any supporting venue or topic information must remain distinct from the evidence for the primary label.

## Rationale structure

Use three short elements:

1. What is the primary contribution?
2. What collection, construction, analysis or synthesis activity establishes it?
3. Why is the nearest alternative less appropriate?

Add a genuine page or section reference and an evidence paraphrase. A paraphrase is not a fabricated quotation. Do not use a model prediction as a reason for the reference label, and do not see the other reader's decision before saving the raw form.

## Difficult and unresolved records

- **Insufficient evidence:** required material is missing, inaccessible or fails to describe the method well enough. Record access and reporting limitations separately.
- **Out of taxonomy:** the research activity is not represented adequately by any defined category. Escalate rather than force a label to preserve sample size.
- **Multiple contributions:** record components and the dominant-contribution rationale. If dominance is not defensible, retain the unresolved state.
- **Version family:** do not annotate two versions as independent analytical units. Send the candidate family to the steward before split membership is finalized.
- **Reader uncertainty:** record difficulty and an alternative independently. Do not calculate these fields from agreement after the fact.

## Pilot, lock and adjudication

The pilot uses 16 separate records and the predeclared kappa action rule in `PROTOCOL.md`. Review disagreements as rule-boundary problems, not as opportunities to coach one reader to reproduce the other's answers. A material guide revision requires a fresh reliability check and re-review of affected earlier records.

After raw forms are sealed, calculate agreement and only then release disagreements for adjudication. The adjudicator receives evidence and rationales but not predictions. Log retained, changed and unresolved decisions, including evidence-based changes to concordant labels. Preserve both raw inputs and the derived reference, with a guide version and correction trail.

## Worked boundary prompts

A paper builds a new scheduling tool and benchmarks it against existing tools. Ask whether the contribution is principally the new artifact or an empirical finding about scheduling behavior; the mere presence of a benchmark cannot decide.

A paper proposes a governance framework and illustrates it with a short fictional example. An illustration alone is not an evaluated artifact or empirical study; inspect what supports the central framework claim.

A paper synthesizes prior studies but provides no reproducible search or selection account. Review may be the correct primary class, while Systematic Review is not yet justified as the secondary label.

A paper surveys practitioners across organizations to compare adoption patterns. Under the inherited broad empirical category it may be Experimental, but that annotation does not imply randomization, intervention or causal identification.
