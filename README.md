# Master Data Entity Resolution — build log (Day 1)

## What's built so far

| File | What it does |
|---|---|
| `scripts/build_master_catalog.py` | Builds the canonical `master_catalog` table in SQLite. `--mode seed` (used today) generates 30 realistic products with valid checksummed EAN-13 barcodes. `--mode real` calls the actual Open Food Facts API — **run this on your own machine**, not in a network-locked sandbox. |
| `scripts/generate_synthetic_artwork.py` | Renders label-style PNG images standing in for real packaging photos (sandbox can't reach Open Food Facts' image CDN). Swap for real downloaded photos later — nothing downstream changes. |
| `scripts/corrupt_supplier_feed.py` | Generates 60 synthetic "supplier submissions" by corrupting master records (brand abbreviations, truncation, case noise, unit reformatting, occasional barcode keying errors), logging the true `master_id` as ground truth. |
| `scripts/normalize.py` | Parses and normalizes the raw feed — folds case, strips punctuation, converts all quantities to a canonical unit (e.g. `0.030kg` → `30g`, `2x51g` → `102g`). |
| `scripts/ocr_extract.py` | Runs Tesseract OCR over the artwork images and scores the extraction against ground truth. |

## Results from today's run

- Master catalog: 30 products, 30 valid GTINs.
- Synthetic supplier feed: 60 submissions, all with at least one corruption applied (tune the probabilities down in `corrupt_supplier_feed.py` if you want a more realistic "mostly clean" mix).
- Normalization: confirmed working — unit and case differences resolve correctly.
- OCR spot-check on 20 artwork images: **100% brand match, 100% GTIN match, 65% quantity match.**

## Honest finding worth remembering for the interview

The quantity mismatches aren't random — they're a real OCR failure mode: Tesseract is misreading a lowercase "g" as "9" and "x" as other characters at this image's font size (`32.5g` → `32.59`, `4x115g` → `4X1159`). This is exactly the kind of thing the build guide's "validate the vision-extraction step separately" section warned about — a benchmark on the text-matching engine wouldn't have caught this at all. Fixing it for real is a fair next step (larger font / higher-res render, or a vision-LLM instead of Tesseract for messier real packaging) — but the fact that you *found* it by actually spot-checking is the story worth telling, not that OCR was perfect.

## Day 2 (Wednesday) — cross-check, candidate matching, and a way to check it yourself

Yesterday's artwork images (20) and supplier feed (60) were sampled independently, so
nothing lined up product-for-product. First fix: `scripts/build_intake_events.py` pairs
each imaged product with one *freshly* corrupted supplier submission for the *same*
product, plus its OCR reading, into one row per "intake event" — `data/intake_events.csv`
(20 events). That's what makes a real cross-check possible.

| File | What it does |
|---|---|
| `scripts/build_intake_events.py` | Builds the 20 paired intake events (artwork + supplier + master, same product each). |
| `scripts/stage03_cross_check.py` | For each event, compares brand/quantity/gtin pairwise across artwork, supplier, and master — flags agreement/disagreement per field, doesn't try to resolve anything. |
| `scripts/stage04_candidate_match.py` | RapidFuzz + phonetic (jellyfish metaphone) + barcode-closeness scoring to retrieve a ranked top-3 shortlist from `master_catalog`, run once keyed on supplier data and once keyed on artwork data. |
| `test_pipeline.py` | 15 pytest assertions covering every stage — barcode validity, normalization, OCR accuracy floors, cross-check correctness, candidate-match recall, and an adversarial near-duplicate-name case. All 15 pass. |
| `interactive_check.py` | CLI tool — pick an intake event (`python3 interactive_check.py evt_1`, or run with no args for a menu) and see the full 01→04 trace with a verdict at the end. |
| `app.py` | Streamlit dashboard version of the same trace, with the pack image shown alongside the data (`streamlit run app.py`) — built for showing a non-technical viewer. |

### Results from today's run

- **Cross-check** (20 events): 1 full agreement, 16 partial agreement, 3 full
  disagreement. Avg fields wrong vs. master — artwork: 0.55/3, supplier: 1.00/3 (expected,
  since supplier corruption is applied ~50–80% of the time per field by design).
- **Candidate matching**: both supplier-keyed and artwork-keyed retrieval hit **100%
  top-1 accuracy** (20/20) against the 30-product catalog — including on near-duplicate
  pairs like "Dairy Milk Chocolate Bar" vs. "Dairy Milk Buttons" and "Mars Bar" vs.
  "Mars Bar Twin Pack". Honest caveat: 30 products is a small catalog, so this isn't a
  stress test of matching *difficulty* — it confirms the retrieval logic itself is sound.
  The adversarial pytest case (two identical brand+name entries distinguished only by
  GTIN) is the harder scenario, and it also passes.
- **pytest**: 15/15 passed, thresholds set from the numbers actually observed (e.g. OCR
  quantity floor of 50%, not 100% — see yesterday's g/9 finding), not assumed targets.

## Day 3 — back-test, real benchmark, and an interactive operating model

Everything above was re-run from scratch on a different machine (macOS, Tesseract
5.5.3, Python 3.14). Several things did **not** reproduce, and the headline
accuracy claim turned out to be measuring the wrong thing.

### Bugs the back-test found (all fixed, all now regression-tested)

| Bug | Effect | Fix |
|---|---|---|
| `generate_synthetic_artwork.py` hardcoded a Linux-only DejaVu font path inside a bare `except OSError` | Silently fell back to PIL's bitmap font on macOS. **OCR brand accuracy 100% → 45%, GTIN 100% → 5%**, no error raised. Every downstream number collapsed with it. | `scripts/fonts.py` resolves a real TTF cross-platform and **raises** rather than falling back |
| `master_id` is `AUTOINCREMENT`, builder only `DELETE`d rows | SQLite keeps the counter in `sqlite_sequence`, so a second run produced ids 31–60. Broke the `artwork_<master_id>.png` join and falsified the "exactly reproducible" guarantee | reset the sequence + pin ids explicitly |
| artwork dir never cleared between runs | stale images accumulated; `ocr_extract.py` crashed with `KeyError` | clear `artwork_*.png` before rendering |
| barcode regex `\b\d{12,13}\b`, first match wins | a 12-digit truncated OCR read passed as a valid GTIN | EAN-13 **check-digit** validation picks the candidate that actually validates |
| `build_intake_events.py` re-implemented the OCR call inline | improvements to `ocr_extract.py` never reached the events stages 03/04 consume | both call one shared `ocr_fields()` |
| stage 04 scored GTIN as `1 - levenshtein/3` | Catalog GTINs are **sequential**, so every product is within edit distance 2 of its neighbours *by construction*. A supplier `barcode_error` corruption could make the wrong near-duplicate outrank the right product (0.86 vs 0.80) | exact match = 1.0; valid-but-different = 0.0; only checksum-**invalid** reads get graded partial credit |

OCR after the fixes: **100% brand, 70% quantity, 95% GTIN** (quantity now beats the
original 65% — the `g`→`9` failure was largely a render-resolution problem, fixed by
rendering at 2x as the original write-up predicted). Cross-check avg fields wrong vs
master reproduces the documented figures exactly: artwork 0.50/3, supplier 1.00/3.

### The accuracy claim was measuring the wrong thing

The old headline — *100% top-1 on 20 events against a 30-product catalog* — still
holds after the fixes, but it is not evidence the matcher works. It is graded on
data the pipeline corrupted itself, against a catalog small enough that retrieval
is close to trivial. So the engine now also runs against the **Leipzig
Amazon-Google and Abt-Buy** benchmarks — real, human-labelled pairs:

| Dataset | Records | Labelled pairs | Best F1 | Precision | Recall | Top-1 |
|---|---|---|---|---|---|---|
| Amazon-Google | 1,363 × 3,226 | 1,300 | **0.616** | 0.617 | 0.615 | 73.3% |
| Abt-Buy | 1,081 × 1,092 | 1,097 | **0.845** | 0.852 | 0.839 | 85.1% |

That is the number worth citing, and it is a very different story from "100%".

**The 0.5 / 0.2 / 0.3 weighting was a guess, and it's measurably the wrong one.**
Swept against labelled data, it is the *worst* mix tested apart from fuzzy-only:

- Amazon-Google: `0.5/0.2/0.3` → F1 0.546  ·  TF-IDF-led `0.3/0/0.7` → **0.616** (+0.070)
- Abt-Buy: `0.5/0.2/0.3` → F1 0.785  ·  TF-IDF-led → **0.845** (+0.060)
- Phonetic weighting actively *hurts* on Amazon-Google (an even 0.34/0.33/0.33 split scores 0.527)

Phonetic matching earns its place on short brand tokens (`MRS`→`Mars`), which is what
this pipeline's own corruptions generate — but on real product titles it adds noise.
That's a genuine finding about where each signal is and isn't useful, not a tuning detail.

### New in this pass

| File | What it does |
|---|---|
| `scripts/matcher.py` | Source-agnostic three-signal engine: fuzzy + phonetic + **TF-IDF char-ngram**. The char-ngram vector space is the "embedding NN search" stage 04 was specified to have, and doubles as the **blocking** step (4.4M candidate pairs → top-50 shortlist) |
| `scripts/benchmark.py` | Leipzig loaders + P/R/F1 harness, plus a precompute/re-score split so weights can be retuned without recomputing signals |
| `scripts/run_benchmark.py` | CLI threshold + weight sweep → `data/benchmark_results.csv` |
| `app.py` | Rebuilt as a two-tab **operating model**: the synthetic pipeline trace, plus a live benchmark console where weights/threshold are sliders and P/R/F1 update instantly across all 1,363 queries |

```bash
python3 scripts/run_benchmark.py --sweep-weights   # reproduce the table above
streamlit run app.py                               # interactive console
pytest test_pipeline.py -q                         # 23 tests
```

### Stage 04 retuned (production weights changed)

Stage 04 now scores **four** signals instead of three — the TF-IDF char-ngram term the
build guide specified for this stage was simply missing:

```
0.20 text (RapidFuzz)  +  0.15 phonetic (metaphone on brand)
                       +  0.35 TF-IDF char-ngram
                       +  0.30 GTIN
```

Two judgement calls behind those numbers, both settled by measurement rather than by
transplanting the benchmark's answer:

- **The benchmark's weights don't port directly.** Its third signal is TF-IDF; stage 04's
  was GTIN. GTIN keeps its 0.30 share because this catalog *has* barcodes (the benchmark
  datasets don't) and `gtin_similarity()` is now authoritative rather than a proximity guess.
- **Phonetic was kept, not dropped.** The sweep showed global phonetic weighting hurting on
  Amazon-Google — but that penalty is an artifact: the benchmark computes metaphone on
  `manufacturer or name`, and most Amazon/Google rows have an empty manufacturer, so it
  degrades to hashing the first word of a product title. Stage 04 uses a real brand field.
  Measured on this pipeline's own corrupted feed against the full catalog, metaphone-on-brand
  fires 69 times and identifies the correct brand **69/69 — zero false positives**, at both
  short (`MRS`, `KLGS`) and long (`Warburtons`) brand lengths. It stays capped below text and
  vector only because knowing the brand still leaves the SKU ambiguous.

Effect on the 20 events: top-1 stays 20/20, but the **top-1-vs-top-2 confidence margin
improves from 0.452 to 0.532 mean (+18%), better on 19/20 events**. That margin is what
stage 07's routing thresholds will key on, so the win is in separation, not headline accuracy.

## Stage 05 — LLM disambiguation (Claude API)

Built. Stage 04 retrieves a shortlist; stage 05 exists for the cases a similarity
score genuinely can't settle — where the artwork-keyed and supplier-keyed retrievals
name different products, where the top two candidates are separated by almost nothing,
or where the text and barcode evidence contradict each other.

```bash
python3 scripts/run_stage05.py               # offline — shows which rows WOULD escalate, spends nothing
python3 scripts/run_stage05.py --live        # actually calls Claude (needs ANTHROPIC_API_KEY)
python3 scripts/run_stage05.py --live --limit 5   # cap spend
```

Uses `claude-opus-5` with adaptive thinking and a Pydantic structured-output schema,
so the verdict comes back as validated fields rather than JSON scraped out of prose.

**Four rules, all enforced in code and regression-tested:**

| Rule | Why |
|---|---|
| **Lazy invocation** — the LLM is only called when `needs_disambiguation()` flags the row | Cost scales with *difficulty*, not volume. Calling an LLM on every row would be slower and dearer than the matcher it assists |
| **It can only pick from the shortlist** | A hallucinated `master_id` can never enter the catalog; an out-of-list answer is rejected as an error |
| **An LLM-only verdict is capped at 0.75** | Stage 07's auto-merge floor is 0.90, so confident-sounding reasoning on weak retrieval evidence can only ever reach "hold for review", never silently write to master |
| **API failure degrades, never crashes** | This stage is an assist, not a dependency — an error returns "no signal" and the batch continues |

The whole stage runs **offline by default**: with no API key it uses `NullDisambiguator`,
which records that a row *would* have escalated without calling anything. All 6 stage-05
tests use an injected fake client, so the suite needs no key and stays deterministic.

**Honest caveat: on the current data this stage has nothing to do.** After the stage-04
retuning, 0 of 20 events trigger it — retrieval is decisive on every row. The trigger
logic and the Claude call path are both tested, but they have not been exercised on a
naturally-occurring ambiguous row, because this 30-product catalog doesn't produce one.
The case it's built for is real but currently synthetic — the mis-keyed-barcode scenario
where retrieval ranks "Mars Bar Twin Pack" (0.86) above the true "Mars Bar" (0.80) while
both quantity readings say 51g. Run `--force-all` to see the stage work on every row.

## Stages 06–08 — barcode verify, routing, resolve + learn

The decision path. Run the whole thing with:

```bash
python3 scripts/run_pipeline.py --demo-alias
```

**Stage 06 — three-way barcode verification.** Previously folded into stage 03's
field loop, which treated a GTIN as just another string. Now its own stage, adding
three things a string compare throws away: check-digit validation (so a *corrupt
reading* is distinguishable from *a valid barcode for a different product* — they
look identical to `==` but mean opposite things), agreement *patterns* rather than a
boolean, and recording **which** two of three agree. Current run: 11/20 all-three-agree,
9/20 two-of-three.

**Stage 07 — confidence and routing.** Combines the three Fig. 2 signals
(0.25 source agreement / 0.35 text match / 0.40 barcode) into one score, then routes
on the guide's bands: ≥0.90 auto-merge, 0.60–0.90 hold, <0.60 reject. Two deliberate
design choices:

- The text signal scores the **margin** between top-1 and top-2, not the winner's
  absolute score. An absolute score says "this looks like a product"; the margin says
  "and it doesn't look like any *other* product" — the thing that matters when writing
  to a master catalog.
- The weighted sum is bounded by explicit **overrides**, because a plain average is
  exactly the "papering over" Fig. 2 warns against. A three-way barcode mismatch, a
  near-tie in retrieval, or an LLM-only decision each cap the score regardless of the
  arithmetic.

**Stage 08 — resolve, log, learn.** SQLite audit log + alias cache. The log preserves
the distinction the guide insists on: a **supplier data-entry error** (stage 03, pack
disagrees with form) is a different problem with a different owner than a **resolution
ambiguity** (stage 07, we can't tell which product). Collapsing them into "failed"
gives you a metric instead of an operational signal. The alias cache stores **only
human-approved** corrections — never the pipeline's own auto-merges, since learning
from unreviewed output is how a resolver drifts: one wrong merge becomes a permanent
"fact" that raises confidence on the next identical wrong input.

### Current end-to-end result (20 events)

| Route | Count | Correct |
|---|---|---|
| auto-merge | 9/20 | 9/9 |
| hold for review | 9/20 | 9/9 |
| reject | 2/20 | 2/2 |
| **incorrect auto-merges** | **0** | — |

The learning loop, demonstrated: `evt_18` holds at 0.833 → a human approves
`Danone SA Actimel Original → Actimel Original` → the identical input resubmitted
**auto-merges at 0.933**.

**A correction worth recording.** The first end-to-end run auto-merged *nothing* —
all 20 held. The cause was mine: I had made any artwork/supplier disagreement a hard
veto, but Fig. 2 treats that as one of three weighted *signals* and specifies only the
three-way barcode mismatch as a hard override. Since the synthetic feed corrupts 100%
of rows, a blanket veto made the auto-merge lane unreachable. Now only an **identity-field**
(barcode) conflict blocks: a brand typed `Mars Inc` against a pack reading `Mars` is
cosmetic, while two different valid barcodes for one pack means a source is describing
another product. A guardrail that blocks everything isn't a guardrail, it's an off switch.

## Real Data Validation (end-to-end on Leipzig)

The pipeline runs stages 03–08 on real e-commerce product pairs from the Leipzig
benchmarks (2010). Each run writes to its own directory, so runs cannot overwrite
each other:

```bash
python3 scripts/build_master_catalog.py  --mode leipzig --dataset abt-buy
python3 scripts/build_intake_events.py   --mode leipzig --dataset abt-buy
python3 scripts/stage03_cross_check.py   --mode leipzig --dataset abt-buy
python3 scripts/stage04_candidate_match.py --mode leipzig --dataset abt-buy
python3 scripts/run_pipeline.py          --mode leipzig --dataset abt-buy
```

**Results (text signals only — these catalogs carry no barcodes and no photos):**

| | Abt–Buy | Amazon–Google |
|---|---|---|
| Submissions | 1,092 | 1,291 |
| Catalog size | 1,081 | 1,363 |
| **Top-1 accuracy** | **87.7%** (958) | **80.2%** (1,035) |
| Top-3 recall | 95% | 95% |
| Surfaced for review | 44 (**44/44 correct**) | 103 (**103/103 correct**) |
| Auto-merged | 0 | 0 |
| Incorrect auto-merges | 0 | 0 |

### Correction: an earlier version of this table claimed 100%

It was a ground-truth leak. `benchmark_to_intake()` copied the catalog record's
own GTIN into the supplier submission, so `gtin_similarity()` scored an exact
barcode match against exactly one row — the correct one — on every event, worth
0.30 of the combined score. The pipeline was being handed the answer.

The measurement that settles it, on a consistent build:

| | Top-1 |
|---|---|
| Supplier submits master's own GTIN (the leak) | 1092/1092 — 100.0% |
| GTIN withheld — real accuracy | 958/1092 — **87.7%** |

87.7% independently corroborates the 85.1% top-1 the benchmark console computes
for this dataset by a separate code path. The leak is now closed (Leipzig events
carry no barcode, which is truthful to the source data) and guarded by
`test_no_supplier_field_is_a_perfect_copy_of_the_master`.

Worth noting how quiet the failure was: every accuracy number went **up**, all
tests stayed green, and the dashboard looked healthier. Nothing about a leak
announces itself, which is why the guard has to be an explicit invariant rather
than a hoped-for smell.

### Why nothing auto-merges, and why that is correct

With no barcode and no packaging photo, exactly one signal is observable. The
router refuses to auto-merge on a single uncorroborated signal, however strong —
a perfect text match with nothing to check it against is precisely the case that
merges two different products with similar names.

Note what it does **not** say: every one of the 44 rows it surfaced for review was
a correct match. The confidence score separates cleanly (AUC 0.907 for the text
margin). It is the *evidence* that is thin, not the matching.

## Calibrated confidence

The routing bands are stated as confidence — auto-merge at ≥0.90, review at
0.60–0.90. That only means something if 0.90 really is "90% of rows scored this
way are correct". It wasn't: confidence was `margin / MARGIN_SATURATION`, and
that constant was set against the 30-product catalog. On 1,081 products the
median margin for a *correct* match is 0.078, so the mapping squashed almost
everything to the bottom of its range.

So the mapping is now **fitted on labelled pairs** rather than guessed.
Platt scaling (a 2-parameter logistic, hand-rolled in `scripts/calibration.py`
so the numbers deciding every threshold are auditable) against isotonic
regression, both scored on data they never saw:

| fit on half of Abt–Buy, score the other half | Brier ↓ | log loss ↓ | ECE ↓ | worst bin ↓ | AUC |
|---|---|---|---|---|---|
| `margin / 0.40` (previous) | 0.530 | 1.544 | 0.656 | 0.794 | 0.908 |
| **Platt (fitted)** | **0.076** | **0.247** | **0.036** | **0.137** | 0.908 |
| Isotonic (fitted) | 0.077 | 0.296 | 0.039 | 0.667 | 0.902 |

AUC is unchanged by design — calibration is monotone, so it cannot alter the
ranking, only what the numbers *mean*. Platt is chosen over isotonic on the
worst-bin figure (0.137 vs 0.667): isotonic follows noise in the sparse tail,
which is exactly where an overconfident band does damage.

### Calibration does not fully transfer between catalogs

This is the question that decides whether the fix is real or just a better-dressed
version of the same mistake. Fitting a curve on Abt–Buy and applying it to
Amazon–Google:

| scored on all of Amazon–Google | Brier ↓ | ECE ↓ |
|---|---|---|
| `margin / 0.40` | 0.452 | 0.557 |
| Platt fitted on **Abt–Buy** | 0.132 | 0.078 |
| Platt fitted on **Amazon–Google** | **0.119** | **0.026** |

Far better than the constant, but measurably overconfident — it claims 0.65 where
the true rate is 0.49. The two fitted slopes differ by 2.7× (63.3 vs 23.4).

So **calibration is per-catalog**, stored as `calibration.json` inside each run
directory. A run without one falls back to the uncalibrated mapping rather than
borrowing another catalog's curve. Shipping one global curve would have been the
original `MARGIN_SATURATION` error with more decimal places.

```bash
python3 scripts/run_calibration.py               # full report
python3 scripts/run_calibration.py --fit abt-buy # fit and save for that catalog
```

### What it changes operationally

Held-out half of Abt–Buy, with stage 07's corroboration rule in force:

| | review queue | rejected | correct matches thrown away |
|---|---|---|---|
| `margin / 0.40` | 24 (100% correct) | 522 | **451** |
| Platt calibrated | 947 (94.1% correct) | 145 | **26** |

**425 correct matches rescued from the reject pile.** Rejection now means "we have
reason to think this doesn't match", not "we couldn't tell".

### The counterfactual worth reading

Disable the corroboration rule and let the calibrated 0.90 band merge, and 398
rows auto-merge at 96.7% precision — **13 wrong records written into the master
catalog.** The band is behaving exactly as specified; 90% is simply not a safe bar
for an unreviewed write. Calibration is what made that visible: an uncalibrated
score can't tell you what your threshold costs.

### What this cannot tell you

Every labelled pair in these benchmarks is a *positive* — the gold mapping only
lists submissions that do have a catalog match. So the fitted probability is
P(correct | a match exists), and **the reject band is never exercised against
genuine no-match submissions**, which a real supplier feed is full of. That is the
next gap, not a solved problem.

### Still honestly outstanding
- `--mode real` (live Open Food Facts) and real product photos remain **unrun**.
- Text↔barcode conflict detection is implemented and flagged, but nothing *routes* on it
  yet — that's stage 07.

### Not done yet

- LLM disambiguation (stage 05) — the tie-breaker for the "only one source resolved
  correctly" case the interactive tool already flags when it happens.
- Confidence scoring / auto-merge / hold-for-review / reject routing.
- Swapping in the real Open Food Facts catalog (`--mode real`) and real photos once
  running outside this sandbox — the bigger, messier catalog this pipeline should
  ultimately be stress-tested against.
