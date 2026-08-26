# Master Data Entity Resolution — 48-Hour Sprint Plan

## Goal
Transform portfolio project into production-ready system with real data, documented case study, and deployment-ready infrastructure.

---

## Day 1 (Hours 0-24)

### Morning (0-8h) — Real Data Integration
**Goal:** Replace synthetic benchmark data with Open Food Facts (food products focus)

- [ ] **Stage 01 — Real OCR on actual packaging**
  - Download 50–100 random product images from Open Food Facts image API
  - Re-run OCR on real packaging (not synthetic renders)
  - Measure accuracy drop vs synthetic (expect 70–80% from 95%)
  - Document failure modes: blurry angles, reflections, foreign text

- [ ] **Stage 02–04 — Real catalog + supplier feed**
  - Pull 500–1000 real food products from OFF API (GTIN, brand, name, category)
  - Simulate supplier submissions by:
    - Taking 200 products as "incoming submissions"
    - Adding realistic noise (OCR errors, abbreviations, missing fields)
  - Measure text matching accuracy on real data vs Leipzig benchmarks
  - Expected: 85–92% top-1 accuracy (vs 100% on 30-row synthetic)

**Deliverable:** `scripts/load_open_food_facts.py` with `--mode real` support

---

### Afternoon (8-16h) — Production Readiness
**Goal:** Make system deployable and scalable

- [ ] **Database optimization**
  - Add indices on master_catalog (brand, product_name, gtin)
  - Implement connection pooling for stage 04 queries
  - Benchmark: time to match 100 submissions (target: <2s)

- [ ] **Error handling & graceful degradation**
  - Stage 04: fallback when TF-IDF vectorizer fails
  - Stage 05: timeout handling for LLM API (return "no signal" at 10s)
  - Stage 06–07: handle missing barcodes (already done)
  - Add structured logging (not just print statements)

- [ ] **Configuration management**
  - Move hardcoded thresholds to `config.yaml`:
    - Stage 04: decision threshold (0.45)
    - Stage 07: routing bands (0.90 / 0.60)
    - Stage 05: LLM timeout, max tokens
  - Add `--config` flag to all runners

**Deliverable:** `config.yaml` + updated runners + logging setup

---

### Evening (16-24h) — Dashboard Enhancements
**Goal:** Make UI production-grade and more informative

- [ ] **Add real-time performance metrics to Streamlit**
  - Timing: OCR latency, match latency, LLM latency per event
  - Cache hit rate (stage 08 alias cache)
  - Signal distribution (why are certain routes common?)

- [ ] **Batch mode for dashboard**
  - Upload CSV of incoming submissions
  - Run pipeline on all, show summary table
  - Export routing decisions + confidence scores

- [ ] **A/B testing UI**
  - Side-by-side: current weights vs alternative weights
  - Show how changing one signal affects routing

**Deliverable:** Enhanced `app.py` with real-time metrics and batch mode

---

## Day 2 (Hours 24-48)

### Morning (24-32h) — Case Study & Documentation
**Goal:** Create compelling portfolio narrative

- [ ] **Write case study document** (`CASE_STUDY.md`)
  - **Problem:** Why entity resolution matters for retail
  - **Approach:** 8-stage pipeline, why each stage exists
  - **Validation:** Real data results vs synthetic
  - **Honest findings:** What works, what doesn't, why
  - **Business impact:** Cost/accuracy tradeoff, decision routing

- [ ] **Create architecture diagram** (SVG or Mermaid)
  - Flow: OCR → Normalize → Cross-check → Match → Barcode → Confidence → Route → Learn
  - Signal flow in stage 04 (4 weighted inputs)
  - Decision tree in stage 07 (confidence + overrides)

- [ ] **README improvements**
  - Add quick-start (3 commands to run)
  - Add system requirements (Python 3.11+, Tesseract, Claude API key)
  - Add deployment section (how to run in production)

**Deliverable:** `CASE_STUDY.md`, architecture diagrams, updated README

---

### Afternoon (32-40h) — Benchmarking & Performance
**Goal:** Provide evidence the system is production-ready

- [ ] **Comprehensive benchmark report**
  - Text matching: accuracy on real OFF data + Leipzig
  - Latency: p50/p95/p99 for each stage
  - Throughput: events/second on typical hardware
  - Memory: peak usage for 1K product catalog
  - Compare: synthetic vs real data performance

- [ ] **Cost analysis**
  - Stage 05 LLM usage: tokens per event, cost per 1000 events
  - Recommend: when to disable stage 05 (cost vs accuracy)
  - Break-even: when is human review cheaper than LLM?

- [ ] **Failure analysis**
  - On real data: which 15% of cases fail and why?
  - Root cause: OCR errors, ambiguous product names, missing signals
  - Recommendations: what would fix each failure class

**Deliverable:** `PERFORMANCE_REPORT.md` with tables, charts, recommendations

---

### Evening (40-48h) — Final Polish & Deploy
**Goal:** Publish production-ready project

- [ ] **Code quality checklist**
  - [ ] 50+ tests covering all 8 stages
  - [ ] Type hints on all functions
  - [ ] Docstrings for public APIs
  - [ ] Remove debug prints and TODO comments
  - [ ] Run: `black`, `isort`, `pylint` on all scripts

- [ ] **CI/CD setup** (GitHub Actions)
  - Run tests on every push
  - Lint and type-check
  - Optional: build Docker image

- [ ] **Final commit + tag**
  - Commit all documentation and performance reports
  - Create git tag: `v1.0-production`
  - Push to GitHub

- [ ] **GitHub profile polish**
  - Add README to profile (link to project)
  - Pin this repository
  - Add topics: `entity-resolution`, `data-matching`, `machine-learning`, `retail`

**Deliverable:** Production-ready codebase, all tests passing, GitHub Actions CI running

---

## Success Criteria (48h)

- [ ] Real data validation (OFF API + real OCR)
- [ ] Production latency benchmarks (all stages <100ms p99)
- [ ] Case study document (compelling narrative)
- [ ] Honest failure analysis (what doesn't work and why)
- [ ] GitHub CI/CD pipeline
- [ ] All tests passing
- [ ] README with 3-command quick start

## Optional Stretch Goals (if time permits)

- [ ] Docker containerization
- [ ] AWS Lambda deployment skeleton
- [ ] Multi-threaded stage 04 for batch mode
- [ ] Prometheus metrics export for monitoring
- [ ] Interactive notebook tutorial (Jupyter)

---

## Resource Needs

- **Claude API** (stage 05 testing): ~$5 budget for 1000 events
- **Open Food Facts API** (free): rate limit ~10 req/sec
- **GitHub Actions** (free): 2000 minutes/month included
- **Time**: 48 hours focused work

---

## Why This Matters

After 48h, you'll have:
1. **Proof it works on real data** (not just synthetic)
2. **Honest performance metrics** (no hiding the hard parts)
3. **Business narrative** (why this matters for retail)
4. **Production readiness** (deployable to actual system)
5. **Strong portfolio piece** (demonstrates full-stack ML + engineering)

This is the difference between "I built a matcher" and "I built a production system, measured it honestly, and documented what works and what doesn't."
