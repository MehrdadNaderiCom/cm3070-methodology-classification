"""Generate every report figure from frozen output tables."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from study.core import sha256
OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)
S = ROOT / "outputs/scenario"
M = ROOT / "outputs/measured"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
                     "axes.labelsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "pdf.fonttype": 42, "figure.dpi": 180,
                     "axes.spines.top": False, "axes.spines.right": False})
TEAL, RUST, INK, GRAY = "#20808D", "#A84B2F", "#28251D", "#7A7974"
MANIFEST = []


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def finish(fig, n, name, source, caption, dpi=None):
    stem = f"fig_{n:02d}_{name}"
    png_kw = {"bbox_inches": "tight", "facecolor": "white"}
    if dpi is not None:
        png_kw["dpi"] = dpi
    fig.savefig(OUT / f"{stem}.png", **png_kw)
    # The CreationDate is pinned to a fixed value so the figure PDFs are
    # byte-for-byte reproducible across runs. This is an intentional
    # reproducibility choice, not a claim that the file was created at that time;
    # the real generation time is recorded separately in figure_manifest.csv.
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white",
                metadata={"Title": caption, "Author": "Mehrdad Naderi",
                          "CreationDate": datetime(2026, 9, 15, 8, 40, 0, tzinfo=timezone.utc)})
    plt.close(fig)
    MANIFEST.append({"figure": n, "name": stem, "source": source,
                     "generator": "scripts/build_figures.py",
                     "generated_utc": datetime.now(timezone.utc).isoformat(),
                     "sha256_pdf": sha256(OUT / f"{stem}.pdf"), "caption": caption})


def flow(n, name, heading, boxes, caption):
    count = len(boxes)
    fig, ax = plt.subplots(figsize=(9.6, 3.7))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.45)
    ax.set_title(heading, loc="left", pad=8, fontsize=12, color=INK)
    left, right, gap = 0.08, 9.92, 0.42
    width = (right - left - gap * (count - 1)) / count
    y0, height = 0.28, 2.42
    for i, (title, detail) in enumerate(boxes):
        x = left + i * (width + gap)
        ax.add_patch(FancyBboxPatch(
            (x, y0), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.07",
            facecolor="#F7F6F2", edgecolor=TEAL, linewidth=1.35,
        ))
        ax.text(x + width / 2, y0 + height - 0.40, title,
                ha="center", va="center",
                fontsize=10 if len(title) > 16 else 11, weight="bold", color=INK)
        ax.text(x + width / 2, y0 + 0.92, textwrap.fill(detail, 16),
                ha="center", va="center", fontsize=9, linespacing=1.45, color=INK)
        if i < count - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.86, y0 + height / 2),
                xytext=(x + width + gap * 0.14, y0 + height / 2),
                arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": 1.2, "mutation_scale": 11},
                clip_on=False,
            )
    finish(fig, n, name, "docs/PROTOCOL.md", caption)


def confusion(n, name, file, title, xlabel, caption):
    data = pd.read_csv(S / file, index_col=0)
    names = ["Case study", "Conceptual", "Design /\nengineering", "Experimental", "Review"]
    fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
    ax.imshow(data.to_numpy(), cmap="Blues", vmin=0)
    ax.set_xticks(range(5), ["Case Study", "Conceptual", "Design", "Experimental", "Review"])
    ax.set_yticks(range(5), names)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlabel(xlabel); ax.set_ylabel("Reference / row label")
    ax.set_title(title, loc="left", pad=13)
    maximum = data.to_numpy().max()
    for i in range(5):
        for j in range(5):
            value = int(data.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=11,
                    color="white" if value > maximum * .55 else INK)
    finish(fig, n, name, f"outputs/scenario/{file}", caption)


def main():
    _root = Path(__file__).resolve().parents[1]
    _need = [
        'data/study_records/adjudicated_labels.csv',
        'legacy/data/processed/corpus.csv',
    ]
    _miss = [p for p in _need if not (_root / p).exists()]
    if _miss:
        raise SystemExit('Private-data command. Missing:\n' + '\n'.join(_miss))

    flow(1, "workflow", "Support a review decision; do not automate exclusion",
         [("Researcher", "Screen candidate titles and abstracts"),
          ("Classifier", "Return ranked labels and a prediction set"),
          ("Reviewer", "Inspect ambiguous cases and record rationale"),
          ("Review record", "Keep the paper and decision traceable")],
         "Figure 1. The review workflow preserves a human inclusion decision.")
    flow(2, "architecture", "Separate point prediction from the review layer",
         [("Input", "CSV or JSON; size, type and ID checks"),
          ("Point model", "Fold-local features and frozen classifier"),
          ("Set layer", "Calibration-only quantile and fixed class order"),
          ("Output", "Label, score, set, review flag and model hash")],
         "Figure 2. Four independent contracts form the classification pipeline.")
    flow(3, "provenance", "Different evidence pools answer different questions",
         [("Pilot: 16", "Outside every analytical pool"),
          ("Development: 230", "Reference audit and model selection"),
          ("Calibration: 60", "30 per external domain; set threshold only"),
          ("Test: 120", "60 per external domain; final evaluation")],
         "Figure 3. The 410 analytical records and 16 pilot records have disjoint roles.")
    flow(4, "taxonomy", "Read the research activity before naming the method",
         [("Evidence", "Methods, collection, analysis and evaluation"),
          ("Contribution", "Identify what the study actually establishes"),
          ("Boundary", "Separate primary contribution from support"),
          ("Decision", "Five-class label or an explicit unresolved state")],
         "Figure 4. Taxonomy decisions use evidence and preserve unresolved cases.")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.axis("off")
    table = ax.table(cellText=[
        ["Raw annotation", "Text only", "No labels", "No predictions", "All IDs"],
        ["Dev release", "Own labels", "Dev labels", "Rationales", "Split map"],
        ["Cal release", "Unchanged", "Cal labels", "Cal rationales", "Test sealed"],
        ["Test release", "Unchanged", "Frozen outputs", "Test rationales", "Release test"],
    ], colLabels=["Gate", "Readers\nA/B", "Modeller", "Adjudicator", "Steward"],
        cellLoc="left", loc="center", colWidths=[.24, .18, .2, .2, .18])
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 3.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D4D1CA")
        if row:
            cell.get_text().set_text(textwrap.fill(cell.get_text().get_text(),
                                                 12 if col == 0 else 10,
                                                 break_long_words=False, break_on_hyphens=False))
        if row == 0: cell.set_facecolor("#E2EEEF"); cell.set_text_props(weight="bold")
    ax.set_title("Role separation prevents prediction-informed annotation", loc="left", pad=8)
    finish(fig, 5, "unblinding", "docs/PROTOCOL.md",
           "Figure 5. The permission matrix separates information access at each release gate.",
           dpi=300)
    confusion(6, "agreement", "development_agreement_confusion.csv",
              "Development agreement is measured before adjudication", "Reader B",
              "Figure 6. Reader A versus reader B on 230 development records; cells show counts.")
    lv = load(S / "label_validity_summary.json")
    confusion(7, "reference_change", "legacy_transition_matrix.csv",
              "A changed reference exposes errors hidden by the old labels", "Adjudicated label",
              f"Figure 7. Legacy-to-adjudicated transitions on the 229 development records that "
              f"carried a legacy label; 100 labels change between existing labels and one "
              f"previously missing label is completed ({lv['changed']} label records affected in total).")
    comp = pd.read_csv(S / "development_comparison.csv")
    fig, ax = plt.subplots(figsize=(7, 3.3), layout="constrained")
    for i, row in comp.iterrows():
        ci = json.loads(row.macro_f1_ci.replace("'", '"'))
        ax.errorbar(row.macro_f1, i, xerr=[[row.macro_f1-ci[0]], [ci[1]-row.macro_f1]],
                    fmt="o", capsize=4, color=TEAL if row.model == comp.sort_values("macro_f1").model.iloc[-1] else GRAY)
        ax.text(.99, i, f"{row.macro_f1:.3f}", va="center", ha="right")
    ax.set_yticks(range(len(comp)), comp.model); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("Macro-F1 with document-bootstrap 95% interval")
    ax.set_title("Candidate comparison on development records", loc="left")
    finish(fig, 8, "development", "outputs/scenario/development_comparison.csv",
           "Figure 8. Development comparison on 230 records; intervals use 10,000 document resamples.")
    results = load(S / "confirmatory_test_results.json")
    model_name = results["selected_model"]
    selected = results["models"][model_name]
    data = [("All test records", selected), *results["domains"].items()]
    fig, ax = plt.subplots(figsize=(7.5, 3.15), layout="constrained")
    for i, (name, r) in enumerate(data):
        ci = r["macro_f1_ci"]; value = r["macro_f1"]
        ax.errorbar(value, i, xerr=[[value-ci[0]], [ci[1]-value]], fmt="o", capsize=4, color=TEAL)
        ax.text(ci[1] + 0.02, i, f"n={r['n']}  {value:.3f}", va="center", ha="left")
    ax.set_yticks(range(3), [n for n, _ in data]); ax.invert_yaxis(); ax.set_xlim(0, 1.08)
    ax.set_xlabel("Macro-F1 with document-bootstrap 95% interval")
    ax.set_title("External performance varies across the two domains", loc="left")
    n_test = int(results["models"][model_name]["n"])
    finish(fig, 9, "external", "outputs/scenario/confirmatory_test_results.json",
           f"Figure 9. Selected {model_name} performance on {n_test} valid test records and the two domains.")
    confusion(10, "test_errors", "test_confusion.csv",
              "Errors remain concentrated at methodology boundaries", "Predicted label",
              f"Figure 10. External {model_name} confusion matrix; row totals give class support.")
    c = results["conformal"]
    fig, axs = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
    # Plot every set size from 0 to the maximum observed (0..5). Using the count
    # of non-zero entries as the range would drop the largest group (size 5).
    max_size = max(int(k) for k in c["set_size_counts"])
    sizes = np.array([c["set_size_counts"][str(i)] for i in range(max_size + 1)])
    assert int(sizes.sum()) == c["n"], "set-size bar heights must sum to n"
    axs[0].bar(range(max_size + 1), sizes, color=TEAL)
    axs[0].set_xticks(range(max_size + 1)); axs[0].set_xlabel("Prediction-set size"); axs[0].set_ylabel("Records")
    for i, v in enumerate(sizes): axs[0].text(i, v + 1, str(v), ha="center")
    axs[0].set_ylim(0, max(sizes) * 1.18)
    axs[0].set_title(f"{c['singleton_n']} singletons; {c['review_n']} reviews", loc="left", fontsize=11)
    vals = [c["errors_in_review"], c["top1_errors"]-c["errors_in_review"]]
    axs[1].bar(["In review", "Outside review"], vals, color=[TEAL, RUST])
    for i, v in enumerate(vals): axs[1].text(i, v + .4, str(v), ha="center")
    axs[1].set_ylim(0, max(vals) * 1.3); axs[1].set_ylabel("Point-model errors")
    axs[1].set_title(f"{c['top1_errors'] - c['errors_in_review']} errors escape the flag", loc="left", fontsize=11)
    finish(fig, 11, "review", "outputs/scenario/confirmatory_test_results.json",
           f"Figure 11. Review flags capture {c['errors_in_review']} of {c['top1_errors']} top-1 errors; coverage is {c['coverage_count']} of {c['n']}, not perfect safety.")
    timings = pd.read_csv(M / "latency_raw.csv")
    cost = load(M / "cost_summary.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.15), layout="constrained")
    axs[0].hist(timings.latency_ms, bins=16, color=TEAL, edgecolor="white")
    axs[0].axvline(5.0, color=GRAY, ls="-", label="5 ms budget")
    axs[0].axvline(cost["median_ms"], color=INK, ls="--", label=f"Median {cost['median_ms']:.2f} ms")
    axs[0].axvline(cost["p95_ms"], color=RUST, ls=":", label=f"P95 {cost['p95_ms']:.2f} ms")
    axs[0].set_xlim(0, 5.2)
    axs[0].set_xlabel("Latency (ms), full range")
    axs[0].set_ylabel("Calls")
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("All calls sit far below 5 ms", loc="left", fontsize=11)
    axs[1].hist(timings.latency_ms, bins=14, range=(0.5, 0.8), color=TEAL, edgecolor="white")
    axs[1].axvline(cost["median_ms"], color=INK, ls="--")
    axs[1].axvline(cost["p95_ms"], color=RUST, ls=":")
    axs[1].set_xlim(0.5, 0.8)
    axs[1].set_xlabel("Latency (ms), 0.5-0.8 detail")
    axs[1].set_title("Mass of the warm-process times", loc="left", fontsize=11)
    fig.suptitle("Measure a defined execution condition, not an unspecified speed", x=0.01, ha="left", fontsize=12)
    finish(fig, 12, "measured_cost", "outputs/measured/latency_raw.csv",
           "Figure 12. Actual lexical inference timings: 150 calls, 30 ordered records across five randomised repeats.")
    pd.DataFrame(MANIFEST).to_csv(OUT / "figure_manifest.csv", index=False)
    print(f"Generated {len(MANIFEST)} figures as editable-source PNG/PDF pairs.")


if __name__ == "__main__":
    main()
