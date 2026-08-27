# Real Data Validation — Interactive Dashboard

The Streamlit app (`streamlit run app.py`) now has **three tabs** instead of two:

## Tab 1: Pipeline Trace (Synthetic)
- Original tab: traces one synthetic 20-event through stages 01–04
- Shows OCR extraction, normalization, cross-check, and candidate matching
- Artwork image displayed alongside the data
- Good for understanding the flow; validates on self-corrupted data

## Tab 2: Real Data Routing (Leipzig) ← NEW
- Stages 03–08 on **real Leipzig benchmark data** (Abt-Buy or Amazon-Google)
- **Dataset selector**: choose between:
  - Abt-Buy (1,081 master products, 1,092 incoming)
  - Amazon-Google (1,363 master products, 3,226 incoming)
- **Event selector**: pick any event from the chosen benchmark
- **Displays:**
  - **Routing decision** (reject/hold/auto-merge) + confidence score + rationale
  - **Three-way comparison**: what artwork OCR read vs. supplier typed vs. master truth
    - (For Leipzig mode: artwork is empty since there are no real images)
  - **Stage 06 barcode verdict**: all-three-agree / two-of-three / full-mismatch / insufficient
  - **Stage 07 signals**: source agreement + text margin + barcode scores
  - **Overrides applied**: which guardrails fired (if any)
  - **Correctness**: whether this was a true match or not

**Key finding:** Top-1 accuracy is **87.7%** on Abt-Buy (958/1,092) and **80.2%** on Amazon-Google (1,035/1,291), text signals only. Nothing auto-merges, because with no barcode and no packaging photo there is exactly one observable signal and the router refuses to write to the catalog on a single uncorroborated signal. Every row it does surface for review is a correct match.

> An earlier version of this document claimed 100%. That was a ground-truth leak: the supplier submission carried the master record's own GTIN. See README for the full correction.

## Tab 3: Benchmark Console (Stage 04 Tuning)
- Original tab: live weight sliders on Leipzig text-matching engine
- Now labeled "Benchmark console (stage 04 tuning)" for clarity
- 1,363 or 3,226 real product pairs with live F1/precision/recall metrics

---

## How to populate Tab 2 with your choice of benchmark

Run the pipeline for the dataset you want:

```bash
# For Abt-Buy (1,081 products)
python3 scripts/build_master_catalog.py --mode leipzig --dataset abt-buy
python3 scripts/build_intake_events.py --mode leipzig --dataset abt-buy
python3 scripts/stage03_cross_check.py --mode leipzig --dataset abt-buy
python3 scripts/stage04_candidate_match.py --mode leipzig --dataset abt-buy
python3 scripts/run_pipeline.py --mode leipzig --dataset abt-buy
streamlit run app.py
```

or for Amazon-Google (1,363 products):

```bash
python3 scripts/build_master_catalog.py --mode leipzig --dataset amazon-google
python3 scripts/build_intake_events.py --mode leipzig --dataset amazon-google
python3 scripts/stage03_cross_check.py --mode leipzig --dataset amazon-google
python3 scripts/stage04_candidate_match.py --mode leipzig --dataset amazon-google
python3 scripts/run_pipeline.py --mode leipzig --dataset amazon-google
streamlit run app.py
```

---

## UI Controls

**Dataset Radio Button**: Select Abt-Buy or Amazon-Google

**Event Dropdown**: Pick any event from the selected benchmark. Events are named:
- `abt-buy_0000`, `abt-buy_0001`, ..., `abt-buy_1090`
- `amazon-google_0000`, `amazon-google_0001`, ..., `amazon-google_1362`

**Display Updates** as you change selections:
- Three-way comparison table (what was read/submitted/known)
- Barcode verdict and detail
- Signal scores (source agreement, text margin, barcode)
- Any overrides that fired (e.g., "barcode_three_way_mismatch", "retrieval_ambiguous")
- Correctness indicator (✓ or ✗)

---

## What This Shows

**Before:** "100% accuracy on real data" — which turned out to be the answer key being copied into the supplier submission.

**Now:** "87.7% top-1 on 1,092 real pairs with the leak closed, corroborated by an independently-computed 85.1% from the benchmark console. Nothing auto-merges because one signal corroborates nothing — and every row surfaced for review is correct."

This is portfolio-grade validation: it shows the system works, acknowledges what's missing, and provides reproducible evidence on real data.
