#!/usr/bin/env python3
"""
Interactive CLI checker -- pick one intake event and watch it move through
every stage the pipeline has built so far, with the ground truth shown
alongside so you can judge for yourself whether each stage did its job.

Usage:
    python3 interactive_check.py            # interactive menu
    python3 interactive_check.py evt_1      # jump straight to one event
    python3 interactive_check.py --all      # print every event, no prompts
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def load(name):
    with open(DATA / name, newline="") as f:
        return {r["event_id"]: r for r in csv.DictReader(f)}

def mark(ok) -> str:
    if ok in ("True", True):
        return "\033[92m✓ match\033[0m"
    if ok in ("False", False):
        return "\033[91m✗ MISMATCH\033[0m"
    return str(ok)

def show_event(event_id, events, checks, matches):
    if event_id not in events:
        print(f"No such event: {event_id}. Try one of: {', '.join(sorted(events)[:10])}...")
        return

    e, c, m = events[event_id], checks[event_id], matches[event_id]

    print("\n" + "=" * 72)
    print(f" INTAKE EVENT: {event_id}   (true master_id = {e['true_master_id']})")
    print("=" * 72)

    print(f"\n[image]    {e['image_path']}")

    print("\n--- Stage 01: Artwork OCR extraction ------------------------------")
    print(f"  brand:     {e['artwork_brand']!r}")
    print(f"  name:      {e['artwork_product_name']!r}")
    print(f"  quantity:  {e['artwork_quantity']!r}")
    print(f"  gtin:      {e['artwork_gtin']!r}")

    print("\n--- Stage 02: Supplier submission (raw -> normalized) -------------")
    print(f"  brand:     {e['supplier_raw_brand']!r} -> {e['supplier_norm_brand']!r}")
    print(f"  name:      {e['supplier_raw_product_name']!r} -> {e['supplier_norm_product_name']!r}")
    print(f"  quantity:  {e['supplier_raw_quantity']!r} -> {e['supplier_norm_quantity']!r}")
    print(f"  gtin:      {e['supplier_raw_gtin']!r} -> {e['supplier_norm_gtin']!r}")
    print(f"  corruption applied: {e['supplier_corruption_applied']}")

    print("\n--- Master catalog (ground truth, for scoring only) ----------------")
    print(f"  brand:     {e['master_brand']!r}")
    print(f"  name:      {e['master_product_name']!r}")
    print(f"  quantity:  {e['master_quantity']!r}")
    print(f"  gtin:      {e['master_gtin']!r}")

    print("\n--- Stage 03: Cross-check (3-way agreement) ------------------------")
    for field in ["brand", "quantity", "gtin"]:
        print(f"  {field:<9} artwork-vs-supplier: {mark(c[f'{field}_artwork_vs_supplier'])}"
              f"   artwork-vs-master: {mark(c[f'{field}_artwork_vs_master'])}"
              f"   supplier-vs-master: {mark(c[f'{field}_supplier_vs_master'])}")
    print(f"\n  overall status: {c['cross_check_status'].upper()}  "
          f"(agreement score {c['agreement_score']})")

    print("\n--- Stage 04: Candidate matching against master_catalog ------------")
    sup_hit = "✓" if m["supplier_top1_hit"] == "True" else "✗"
    art_hit = "✓" if m["artwork_top1_hit"] == "True" else "✗"
    print(f"  supplier-keyed top-3: {m['supplier_candidates']}   [{sup_hit} top-1 correct]")
    print(f"  artwork-keyed  top-3: {m['artwork_candidates']}   [{art_hit} top-1 correct]")

    print("\n--- Verdict ----------------------------------------------------------")
    if m["supplier_top1_hit"] == "True" and m["artwork_top1_hit"] == "True":
        print("  \033[92mBoth sources independently resolve to the correct product.\033[0m")
    elif m["supplier_top1_hit"] == "True" or m["artwork_top1_hit"] == "True":
        print("  \033[93mOnly one source resolved correctly on its own -- this is exactly the\033[0m")
        print("  \033[93mcase stage 05 (LLM disambiguation) exists to break the tie.\033[0m")
    else:
        print("  \033[91mNeither source's top pick was correct -- would route to manual review.\033[0m")
    print("=" * 72 + "\n")

def main():
    events = load("intake_events.csv")
    checks = load("cross_check_results.csv")
    matches = load("candidate_match_results.csv")
    ids = sorted(events, key=lambda x: int(x.split("_")[1]))

    args = sys.argv[1:]
    if args and args[0] == "--all":
        for eid in ids:
            show_event(eid, events, checks, matches)
        return
    if args:
        show_event(args[0], events, checks, matches)
        return

    print(f"Loaded {len(ids)} intake events. Type an event id to inspect it,")
    print(f"'list' to see all ids, 'random' for a random one, or 'quit' to exit.")
    print(f"Example ids: {', '.join(ids[:5])}")

    import random
    while True:
        choice = input("\n> ").strip()
        if choice in ("quit", "exit", "q"):
            break
        if choice == "list":
            print(", ".join(ids))
            continue
        if choice == "random":
            choice = random.choice(ids)
        show_event(choice, events, checks, matches)

if __name__ == "__main__":
    main()
