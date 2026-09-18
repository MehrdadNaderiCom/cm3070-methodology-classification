# Documentary workflow casebook

## How to read these cases

The opening scope note in the report governs this file. Each row below is traceable to `outputs/public/test_row_results.csv` in the public tree (or the private scenario table). Actions and mitigations are analytical recommendations, not observed participant responses.

| Case | Reference | Top label | Prediction set | Review flag | Decision problem | Recommended action |
|---|---|---|---|---|---|---|
| EXT130 | Design/Engineering | Experimental | Experimental | No | Wrong singleton creates false reassurance | Read whether the artefact or the measurement is the contribution; permit override despite no flag. |
| EXT100 | Design/Engineering | Review | Review | No | An artefact paper is mistaken for a literature synthesis | Inspect whether a system or method is built rather than surveyed. |
| EXT000 | Design/Engineering | Experimental | All five classes | Yes | Evaluation wording pulls an artefact paper toward Experimental | Check what evidence establishes the contribution; record the boundary rationale. |
| EXT006 | Experimental | Experimental | All five classes | Yes | A correct top label arrives with maximal ambiguity | Confirm the source of the evidence before accepting the label. |
| EXT018 | Experimental | Experimental | Experimental | No | Routine correct singleton | Preserve traceability; do not relabel a singleton as human verified. |

## What can and cannot be concluded

The casebook establishes that the frozen outputs contain both useful review opportunities and residual wrong singletons. It motivates interface requirements: show every set member, separate "not flagged" from "verified," retain overrides, preserve the original output and record the review rationale.

It does not establish demand, task completion time, user acceptance, confidence calibration or improved human accuracy. The oracle figure of 108/111 correct assumes perfect correction of all flagged errors and no introduced errors.

## Later participant-study specification

- **Recruitment:** 12-20 qualified methodology reviewers after a separate 4-6-person usability pilot, subject to actual permission and consent.
- **Conditions:** unaided text review, point-label assistance and prediction-set assistance. Use counterbalanced within-participant order and disjoint, class-balanced item blocks.
- **Outcomes:** per-participant accuracy, wrong acceptance, corrections, time per item, subjective workload and confidence. Record overrides and whether the reference belongs to the displayed set.
- **Analysis:** pair participant-level outcomes, report uncertainty and account for item difficulty. Do not count every response as an independent participant.
- **Stop rules:** prediction exposure in reference creation, unauthorized data access, withdrawn consent or a material interface change during the comparison.
