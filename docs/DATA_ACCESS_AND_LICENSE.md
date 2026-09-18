# Data access and licensing

This public tree redistributes:

- MIT-licensed code;
- authored demo texts under `data/demo/`;
- paper identifiers, titles, DOIs, splits and labels under `data/public/paper_index.csv`;
- aggregate evaluation tables under `outputs/public/`;
- authored figures.

It does **not** redistribute:

- paper abstracts or publisher PDFs;
- raw reader forms or timestamps;
- the archival TF-IDF binary or MPNet embeddings;
- handover, ownership, or permission worksheets;
- the full report PDF and its chapter sources, which are submitted via the course delivery path only.

Ten external identifiers are marked `excluded_from_confirmatory_analysis` because a recorded retrieval clock in the private archive precedes the arXiv publication date. Original clocks were not rewritten. Those ten records (nine test, one calibration) are excluded from the published confirmatory tables. Cue on the remaining 111 test records scores 63 correct (macro-F1 0.532); conformal coverage is 105/111 with 100 review flags after refitting the threshold on 59 calibration records.

Code is MIT-licensed in `LICENSE`.
