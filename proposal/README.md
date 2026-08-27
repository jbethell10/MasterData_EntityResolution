# Proposal document

`mder_proposal.tex` — 3–4 page proposal for the MDER pipeline.

## Before circulating

Edit two commands near the top of the file:

```latex
\newcommand{\orgname}{[Organisation]}
\newcommand{\authorname}{Jacob Bethell}
```

## Compiling

No TeX toolchain is installed on this machine. Two options:

**Overleaf** (nothing to install) — upload `mder_proposal.tex`, it compiles as-is.
Every package used ships in the base TeX Live distribution.

**Locally** — install MacTeX (large) or BasicTeX (small):

```bash
brew install --cask basictex
# then, in a new shell:
sudo tlmgr update --self && sudo tlmgr install booktabs titlesec enumitem
cd proposal && pdflatex mder_proposal.tex && pdflatex mder_proposal.tex
```

Run twice so cross-references resolve.

## Figures quoted, and where they come from

| Figure | Source |
|---|---|
| 87.7% top-1, Abt–Buy (1,081 products) | `scripts/run_pipeline.py --mode leipzig --dataset abt-buy` |
| 80.2% top-1, Amazon–Google (1,363) | same, `--dataset amazon-google` |
| 24% top-1 from photographs (2,256) | `scripts/ocr_extract.py --mode real` + retrieval |
| Brier 0.530→0.076, ECE 0.656→0.036 | `scripts/run_calibration.py` (held out) |
| 425 correct matches rescued | `scripts/run_calibration.py`, Q3 |
| 13 wrong merges if corroboration disabled | `scripts/run_calibration.py`, counterfactual |
| 86.8% EAN-13 / 12.9% EAN-8 | `scripts/fetch_open_food_facts.py` |
| 63 tests | `pytest test_pipeline.py -q` |
