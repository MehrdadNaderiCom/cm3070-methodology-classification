# Reference standard and evaluation protocol

## Roles and permissions

Protocol version: 2026-09-15.

The protocol was executed by two fixed annotators: the author (A) and a colleague (B); a separate adjudicator (C) resolved disagreements after the raw files were sealed. The author also acted as data steward. No participant user study, public release or examiner regrade is asserted by this document.

The steward controls identity mappings, duplicate review and release boundaries. Annotators see document text and the locked guide, but not legacy labels, model predictions, split membership or each other's decisions. The adjudicator receives rationales and relevant full text, never predictions. The modeller sees development labels first, calibration labels only after external predictions are frozen, and test labels only after the review rule and analysis plan are frozen.

Before real annotation begins, record qualifications, permission, data-access terms and conflict checks. If one person occupies incompatible roles, stop rather than claiming blinding that does not exist.

## Corpus and reference design

Use all 230 accessible development papers, 60 external calibration papers and 120 external test papers. Calibration contains 30 AI-adoption and 30 computing-education papers; test contains 60 from each domain. Choose 12-20 separate pilot papers. The supplied worked path uses 16 pilot records.

Deduplicate by stable identifier, DOI, normalized title and reviewed near-duplicate or version-family candidates before sampling. Record one exclusion reason per excluded item. Determine external membership from domain and seed, not observed methodology labels. A matching domain mixture supports, but does not establish, exchangeability.

The original development corpus has 229 populated five-class labels, not 230. It has 230 seven-class labels. Preserve the missing five-class outcome rather than filling it without evidence.

## Annotation procedure

Use `TAXONOMY_GUIDE.md` for the locked class definitions, nearest-alternative boundaries and unresolved-case rules. The guide complements the fields in `data/annotation_form.csv`.

Read Methods, Methodology or Study Design, then collection, analysis, Results and Evaluation. Use the abstract for context. Record the primary contribution, the method used to establish it, the evidence location and a rationale that distinguishes the preferred label from its nearest alternative.

The five-class operational vocabulary is Case Study, Conceptual, Design/Engineering, Experimental and Review. "Experimental" retains the inherited broad empirical definition, including some observational and questionnaire work; its name must not be interpreted as a claim of random allocation or causal identification. Use an ambiguity flag when that breadth obscures an important distinction.

Seven-class labels remain secondary. Distinguish narrative from systematic reviews only when a repeatable search and selection protocol is evidenced. Record mixed-method components before selecting a dominant five-class mapping. Never force an unresolved mixed-method paper into a category merely to preserve the denominator.

The blank form includes document ID, reader role, guide/form version, timestamps, five-class label, seven-class label, discipline, field, opened sections, evidence, page or section, rationale, alternatives, difficulty, out-of-taxonomy and insufficient-evidence flags, access/interruption notes and save timestamp. Evidence paraphrases are in the reader's words; they are not copied abstract sentences.

## Pilot and agreement

Lock the action rule before inspecting pilot results: kappa at least 0.80 permits progression; 0.67-0.79 requires clarification and a fresh check; below 0.67 requires material revision and a new pilot. These are project decisions, not universal validity thresholds.

Compute exact agreement and Wilson intervals, unweighted Cohen's kappa with document-bootstrap intervals, category-specific positive agreement and the confusion matrix before adjudication. AC1 is a sensitivity statistic. Preserve both raw files. Repeat approximately 10% of items per reader in a separate round; report intra-reader results separately.

Resolve disagreements against the locked guide and log the evidence and final reason. If a material guide change is necessary, version the guide and re-review affected earlier records symmetrically. Consensus is not proof of truth; a common reader error may survive disagreement-only adjudication.

## Registered analysis

Methodology5 macro-F1 on the 120-record external test is primary. Majority, cue matching, TF-IDF and MPNet are the four compared candidates; SPECTER2 is omitted from the core because it is optional and does not resolve the reference-standard problem.

Use the same group-disjoint outer folds for all candidates, with inner selection inside each training partition. The real lexical implementation searches C in {0.3, 1.0, 3.0}, holds the TF-IDF specification fixed and validates every realized group split. The worked selection rule selects the candidate with the highest development out-of-fold macro-F1, with ties broken toward the simpler model. No test result can reverse this decision.

Preserve old predictions when comparing two label references. Never retrain between the two evaluations. Compare new candidates only after development annotation is frozen. Full-text-derived reference labels do not change the classifier input, which remains title and abstract.

## Freeze and calibration

Freeze the point model, preprocessing, class order, training IDs, labels, code identity and external predictions before calibration-label release. Split conformal uses score 1 minus the probability assigned to the true class, alpha 0.10 and rank ceil((n+1)(1-alpha)). At n=60 this is the 55th sorted score. Include every class whose score is less than or equal to the threshold.

The bounded probability score uses threshold 1 when the requested order statistic is n+1, yielding the full label set. Empty sets are valid outputs for other thresholds and must be reviewed. The policy flags every set whose size is not one, without substituting the top-ranked class into an empty set.

Hash the conformal configuration separately from the point model. Do not attach a threshold generated from one model or corpus to another.

## Test and uncertainty

Join predictions and reference labels by unique ID with exact set equality. Report the registered primary metric first, then Wilson accuracy intervals, per-class support, confusion counts and domain summaries. Use 10,000 paired domain-stratified bootstrap resamples for the MPNet-minus-TF-IDF difference and one exact McNemar test for top-1 correctness.

Report empirical set coverage with an exact binomial interval, set-size distribution, empty/singleton counts, workload and error capture. A singleton is not an automatic inclusion or exclusion decision. Marginal coverage does not imply coverage for every class, domain, reviewer or individual paper.

## Workflow evidence

Use the documentary fallback. The casebook records a frozen output, the decision it invites, a reasonable reviewer action, a failure mechanism and a mitigation. It does not estimate user time saved, acceptance, workload reduction or improved human accuracy.

For a later permissioned study, use 12-20 qualified participants after a separate 4-6-person usability pilot. Use a counterbalanced within-participant crossover, disjoint balanced item blocks and paired participant-level outcomes. Record incorrect acceptance as well as correction. Do not treat repeated item responses as independent participants.

## Stop rules

Stop at uncertain permission, insufficient pilot reliability, prediction exposure, duplicate contamination, an invalid class contract or an unexplained hash change. Preserve a failed first test run. A mechanical correction requires a deviation record and both old and corrected outputs; a disappointing result does not.
