"""Show frozen Cue review flags using public identifiers only."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASES = ("EXT000", "EXT018", "EXT130")

def main():
    index = pd.read_csv(ROOT / "data/public/paper_index.csv")
    rows = pd.read_csv(ROOT / "outputs/public/test_row_results.csv")
    titles = dict(zip(index.document_id, index.title))
    print("Public demo: Cue review flags (alpha 0.10). A singleton is not verified.\n")
    for cid in CASES:
        r = rows[rows.document_id.eq(cid)]
        if r.empty:
            continue
        r = r.iloc[0]
        print(f"{cid}  {str(titles.get(cid, ''))[:88]}")
        print(f"  reference {r.label}")
        print(f"  top-1     {r.top1}   correct={bool(r.correct)}")
        print(f"  set size  {int(r.set_size)}   review={bool(r.review)}")
        print()
    print("Limitation: 100/111 valid test records are flagged at the frozen rule.")

if __name__ == "__main__":
    main()
