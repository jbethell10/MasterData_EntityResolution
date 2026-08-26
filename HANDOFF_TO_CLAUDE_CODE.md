# MDER Pipeline — Handoff to Claude Code

## What this project is

A **Master Data Entity Resolution (MDER)** pipeline prototype: it reconciles three independent, noisy views of the same retail product — what's printed on the **packaging artwork** (read via OCR), what a **supplier typed** into a submission portal (deliberately corrupted to simulate real-world messiness), and the canonical **master catalog** record — and tries to resolve them to a single confident match.

It was built as a portfolio project for an FDE (Forward Deployed Engineer) application, on **public/synthetic data only** (no proprietary employer data). It was built and run inside a network-sandboxed cloud environment that couldn't reach Open Food Facts or download real product photos, so two deliberate substitutions were made (both documented, both swappable):

- `master_catalog` is a **seed set of 30 real-brand, fictional-SKU products** with correctly checksummed EAN-13 barcodes, not the live Open Food Facts catalog.
- Packaging photos are **synthetically rendered label images** (PIL), not real downloaded photos.

Both scripts that made these substitutions already have a "real" code path written and ready — they just need to run somewhere with open internet access, which is exactly what Claude Code running locally provides.

## Pipeline stages built so far (of 8 designed)

| # | Stage | Script | Status |
|---|---|---|---|
| 00 | Master catalog | `scripts/build_master_catalog.py` | ✅ built (`--mode seed` run; `--mode real` written, untested) |
| — | Synthetic supplier feed | `scripts/corrupt_supplier_feed.py` | ✅ built (60 rows) |
| — | Synthetic artwork images | `scripts/generate_synthetic_artwork.py` | ✅ built (20 images) |
| 01 | Artwork OCR extraction | `scripts/ocr_extract.py` | ✅ built |
| 02 | Supplier normalization | `scripts/normalize.py` | ✅ built |
| — | Paired intake events (artwork + supplier + master, same product) | `scripts/build_intake_events.py` | ✅ built (20 events) |
| 03 | Cross-check (3-way agreement) | `scripts/stage03_cross_check.py` | ✅ built |
| 04 | Candidate matching (RapidFuzz + phonetic + barcode) | `scripts/stage04_candidate_match.py` | ✅ built |
| 05 | LLM disambiguation | — | ❌ not built |
| 06 | Formal 3-way barcode verification | — | ❌ not built (folded informally into stage 03 today) |
| 07 | Confidence scoring / routing (auto-merge / hold / reject) | — | ❌ not built |
| 08 | Resolve + alias-learning cache | — | ❌ not built |

## Verification tooling already built

- `test_pipeline.py` — 15 pytest assertions covering barcode validity, normalization, OCR accuracy floors, cross-check correctness (does it actually flag known corruptions?), candidate-match recall, and one hand-built adversarial case (two catalog entries with identical brand+name, distinguished only by GTIN). **All 15 passed** on the last run in the sandbox.
- `interactive_check.py` — CLI: `python3 interactive_check.py evt_1` prints the full stage 01→04 trace for one event plus a plain-English verdict. Also supports `--all` and an interactive menu with no arguments.
- `app.py` — Streamlit dashboard version of the same trace (`streamlit run app.py`), with the artwork image rendered next to the data.

## Results observed in the sandbox (baseline to compare against)

- **OCR extraction** (20 images): 100% brand match, 100% GTIN match, **65% quantity match**. Root cause found and documented, not hidden: Tesseract misreads lowercase "g" as "9" and "x" as other characters at the rendered label's font size (`32.5g` → `32.59`). A larger font / higher-res render, or a vision-LLM instead of Tesseract, would likely fix this on real packaging.
- **Cross-check** (20 paired events): 1 full agreement, 16 partial agreement, 3 full disagreement. Avg fields wrong vs. master — artwork: 0.55/3, supplier: 1.00/3 (expected, since supplier corruption is injected on ~50–80% of fields by design).
- **Candidate matching**: both supplier-keyed and artwork-keyed retrieval hit **100% top-1 accuracy (20/20)** against the 30-product catalog, including on near-duplicate pairs (e.g. "Dairy Milk Chocolate Bar" vs. "Dairy Milk Buttons"). Honestly caveated: 30 products is a small catalog, so this confirms the retrieval *logic* is sound, not that it's been stress-tested at scale.
- **pytest**: 15/15 passed.

## Known constraints from the sandbox that Claude Code should be aware of

1. `pytesseract` requires the **Tesseract OCR binary** installed on the system (not just the Python package) — check with `tesseract --version`, install via the OS package manager if missing (e.g. `brew install tesseract` on macOS, `apt install tesseract-ocr` on Debian/Ubuntu).
2. All random generation uses fixed seeds (`random.seed(...)` at specific values per script) so results should be **exactly reproducible** on a re-run — any drift is a genuine finding, not noise, and is worth investigating rather than re-running until it matches.
3. `build_master_catalog.py --mode real` and swapping in real Open Food Facts photos were **never actually run** anywhere (the sandbox couldn't reach the network) — treat these as untested code, not verified features, when back-testing.

---

## Prompt to paste into Claude Code

Copy everything below into Claude Code once the project folder is on your machine (or point it at the repo if you've pushed it somewhere).

```
I'm handing you an in-progress Master Data Entity Resolution (MDER) pipeline
prototype, built in a network-sandboxed environment. I want you to back-test
everything that's been built so far — not just confirm it "looks fine," but
actually re-run it, compare the numbers you get against the numbers already
documented, and tell me honestly where reality differs from the documentation.

Read HANDOFF_TO_CLAUDE_CODE.md and README.md first for full context on what
this project is and what's already been claimed.

Then do the following, in order, and give me a written report at the end:

1. ENVIRONMENT CHECK
   - Confirm Python version and that pytesseract, PIL/Pillow, rapidfuzz,
     jellyfish, pytest, streamlit, and pandas are installed (pip install
     whatever's missing).
   - Confirm the Tesseract OCR binary itself is installed and on PATH
     (`tesseract --version`) — this is a system dependency, not just a pip
     package, and OCR results will be silently wrong or the script will
     crash if it's missing or a very different version than what built this.

2. FULL RE-RUN, IN ORDER
   Re-run every script from scratch in this exact order, and capture the
   printed output of each:
     scripts/build_master_catalog.py --mode seed
     scripts/generate_synthetic_artwork.py
     scripts/corrupt_supplier_feed.py
     scripts/normalize.py
     scripts/ocr_extract.py
     scripts/build_intake_events.py
     scripts/stage03_cross_check.py
     scripts/stage04_candidate_match.py
   For each one, compare its printed summary numbers against the numbers in
   README.md / HANDOFF_TO_CLAUDE_CODE.md. Since every script uses a fixed
   random seed, results should reproduce exactly — flag ANY discrepancy
   (even a single row) as a real finding and dig into why, rather than
   assuming it's fine. A likely source of drift is Tesseract OCR accuracy
   if your installed Tesseract version differs from what built this.

3. AUTOMATED TEST SUITE
   Run `pytest test_pipeline.py -v` and report the real pass/fail count.
   All 15 tests passed in the original sandbox run — if any fail here,
   show me the full assertion failure, not just "some tests failed."

4. INTERACTIVE / MANUAL SPOT-CHECK
   Run `python3 interactive_check.py --all` and read through the output.
   Specifically look for:
     - any event where the cross-check status doesn't match what the
       corruption_applied column in data/intake_events.csv would predict
     - any event where candidate matching's top-1 pick is wrong even
       though I've claimed 100% top-1 accuracy — if you find even one,
       tell me immediately, don't average it away
   Then run `streamlit run app.py` and confirm it starts without error
   (curl localhost or check the process, you don't need a browser).

5. CODE-LEVEL REVIEW (not just "does it run")
   Read through scripts/stage03_cross_check.py and scripts/stage04_candidate_match.py
   specifically and look for logic bugs that a passing test suite might not
   catch — e.g., is the cross-check's field-equality function too strict or
   too lenient in a way that would misclassify agreement in a case the
   current 20-event sample doesn't happen to exercise? Is the candidate
   scoring formula's weighting (0.5 text / 0.2 phonetic / 0.3 GTIN) actually
   justified, or just a guess that happened to work on this small catalog?
   Try constructing 2-3 new adversarial test cases beyond the ones already
   in test_pipeline.py and tell me if they break anything.

6. UNTESTED CODE PATHS (only run these if I confirm I want live network
   calls made)
   - scripts/build_master_catalog.py --mode real (hits the live Open Food
     Facts API) has never actually been executed anywhere — ask me before
     running it, since it will attempt real network calls and I want to
     control when that happens.
   - Ask before attempting to download and swap in real product photos.

7. REPORT
   Give me a plain-language summary: what reproduced exactly as documented,
   what didn't (with numbers), any bugs or edge cases you found in step 5,
   and whether you'd trust this pipeline's current results if I were citing
   them in an interview. Don't soften genuine problems to make the project
   look more finished than it is — an honest "here's what's actually shaky"
   is more useful to me than reassurance.
```

---

## Suggested next steps after the back-test (not part of the prompt above — for later)

Once Claude Code confirms the current stages hold up, the remaining build (per the original schedule) is:

- **Stage 05** — LLM disambiguation for the cases where supplier-keyed and artwork-keyed candidate matching disagree (the interactive tool already flags these).
- **Stage 06** — formal three-way barcode verification as its own explicit stage (today it's folded into stage 03's gtin field comparison).
- **Stage 07** — confidence scoring and routing into auto-merge / hold-for-review / reject lanes.
- **Stage 08** — resolve + alias-learning cache (so a corrected match, e.g. "MRS" → "Mars", is remembered for next time).
- Validating stage 04's candidate matching against a real entity-resolution benchmark (Leipzig Abt-Buy or Amazon-Google) rather than just this synthetic catalog, to get a credible accuracy number to cite.
- Running `--mode real` and swapping in real Open Food Facts photos, now that Claude Code has open network access this sandbox didn't.
