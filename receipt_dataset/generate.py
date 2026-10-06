#!/usr/bin/env python3
"""Synthetic dataset generator for receipt OCR text correction.

Each example pairs a clean, realistic receipt (the target) with a copy that
has been corrupted using the kinds of errors OCR engines make on thermal
paper receipts (the input). No third-party dependencies.

Usage:
    python generate.py --out data --train 8000 --val 1000 --test 1000 --seed 42
    python generate.py --granularity line   # one example per receipt line
"""

import argparse
import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------
# Receipt content vocabulary
# --------------------------------------------------------------------------

STORES = {
    "grocery": [
        "FRESH MART", "GREEN VALLEY GROCERS", "SUNRISE FOODS", "CORNER MARKET",
        "HARVEST SUPERMARKET", "MAPLE LEAF GROCERY", "CITY FRESH", "VALUE FOODS",
    ],
    "restaurant": [
        "THE RUSTIC SPOON", "BLUE OCEAN GRILL", "MAMA ROSA'S PIZZERIA",
        "GOLDEN DRAGON", "THE BURGER JOINT", "CAFE LUMIERE", "TACO FIESTA",
    ],
    "coffee": [
        "BEAN THERE CAFE", "DAILY GRIND", "MORNING BREW CO", "ESPRESSO LANE",
    ],
    "pharmacy": [
        "HEALTHPLUS PHARMACY", "CARE DRUG STORE", "WELLNESS RX", "CITY CHEMIST",
    ],
    "hardware": [
        "HANDY HARDWARE", "BUILD IT DEPOT", "TOOLBOX SUPPLY", "ACE FIX-IT",
    ],
    "gas": [
        "QUICKFUEL", "HIGHWAY GAS & GO", "PETRO STOP", "EXPRESS FUEL",
    ],
}

ITEMS = {
    "grocery": [
        ("BANANAS", 0.59, 3.50), ("WHOLE MILK 1GAL", 2.99, 5.49),
        ("LARGE EGGS 12CT", 2.49, 6.99), ("WHITE BREAD", 1.99, 4.29),
        ("CHEDDAR CHEESE", 3.49, 7.99), ("CHICKEN BREAST", 5.99, 14.99),
        ("GROUND BEEF 1LB", 4.99, 8.99), ("APPLES GALA", 1.29, 5.99),
        ("ORANGE JUICE", 3.29, 6.49), ("PASTA PENNE", 0.99, 2.99),
        ("TOMATO SAUCE", 1.49, 3.99), ("BUTTER UNSALTED", 3.99, 6.49),
        ("GREEK YOGURT", 0.99, 6.99), ("BABY SPINACH", 2.99, 4.99),
        ("RICE JASMINE 2LB", 2.99, 5.99), ("COFFEE BEANS", 7.99, 14.99),
        ("PAPER TOWELS", 4.99, 12.99), ("DISH SOAP", 2.49, 4.99),
        ("AVOCADO", 0.99, 2.49), ("POTATO CHIPS", 2.99, 4.99),
        ("SPARKLING WATER", 3.99, 6.99), ("CEREAL OATS", 3.49, 5.99),
    ],
    "restaurant": [
        ("CAESAR SALAD", 8.50, 14.00), ("MARGHERITA PIZZA", 12.00, 18.00),
        ("CHEESEBURGER", 10.50, 16.00), ("FISH & CHIPS", 13.00, 19.00),
        ("FRENCH FRIES", 3.50, 6.00), ("CHICKEN WINGS", 9.00, 15.00),
        ("SPAGHETTI BOLOGNESE", 13.50, 19.50), ("ICED TEA", 2.50, 4.00),
        ("SODA", 2.00, 3.50), ("HOUSE WINE", 7.00, 12.00),
        ("DRAFT BEER", 5.00, 8.50), ("CHOCOLATE CAKE", 6.00, 9.00),
        ("TACOS AL PASTOR", 9.00, 14.00), ("FRIED RICE", 8.00, 12.00),
    ],
    "coffee": [
        ("LATTE 12OZ", 3.75, 5.50), ("CAPPUCCINO", 3.50, 5.25),
        ("AMERICANO", 2.75, 4.25), ("ESPRESSO DBL", 2.50, 3.75),
        ("CHAI LATTE", 4.00, 5.75), ("BLUEBERRY MUFFIN", 2.75, 4.00),
        ("CROISSANT", 2.95, 4.25), ("BAGEL W/ CREAM CHS", 3.25, 4.75),
        ("COLD BREW", 3.95, 5.50), ("OAT MILK ADD", 0.60, 0.90),
    ],
    "pharmacy": [
        ("IBUPROFEN 200MG", 5.99, 11.99), ("VITAMIN C 500MG", 6.49, 12.99),
        ("BANDAGES ASST", 3.99, 7.99), ("TOOTHPASTE", 2.99, 5.99),
        ("SHAMPOO 12OZ", 4.99, 9.99), ("HAND SANITIZER", 1.99, 4.99),
        ("ALLERGY RELIEF", 8.99, 19.99), ("COUGH SYRUP", 7.49, 12.99),
        ("COTTON SWABS", 1.99, 3.99), ("SUNSCREEN SPF50", 8.99, 14.99),
    ],
    "hardware": [
        ("WOOD SCREWS 1LB", 5.99, 9.99), ("HAMMER 16OZ", 12.99, 24.99),
        ("DUCT TAPE", 4.99, 8.99), ("LED BULB 4PK", 7.99, 15.99),
        ("PAINT BRUSH 2IN", 3.99, 8.99), ("EXTENSION CORD", 9.99, 19.99),
        ("SANDPAPER ASST", 4.49, 7.99), ("WD-40 8OZ", 4.99, 7.49),
        ("ZIP TIES 100CT", 3.99, 6.99), ("MEASURING TAPE", 8.99, 16.99),
    ],
    "gas": [
        ("UNLEADED REG", 30.00, 75.00), ("PREMIUM UNL", 40.00, 90.00),
        ("DIESEL", 45.00, 110.00), ("BOTTLED WATER", 1.29, 2.49),
        ("ENERGY DRINK", 2.49, 3.99), ("BEEF JERKY", 4.99, 8.99),
        ("CAR WASH BASIC", 8.00, 12.00), ("CANDY BAR", 1.29, 2.29),
    ],
}

STREETS = ["MAIN ST", "OAK AVE", "ELM ST", "MARKET ST", "PARK BLVD",
           "2ND AVE", "BROADWAY", "LAKE RD", "HIGHLAND DR", "RIVER RD"]
CITIES = [("SPRINGFIELD", "IL"), ("RIVERSIDE", "CA"), ("FRANKLIN", "TN"),
          ("GREENVILLE", "SC"), ("MADISON", "WI"), ("SALEM", "OR"),
          ("AUSTIN", "TX"), ("PORTLAND", "ME"), ("DENVER", "CO")]
CASHIERS = ["JOHN", "MARIA", "ALEX", "PRIYA", "SAM", "LUIS", "KIM", "OMAR"]
CARDS = ["VISA", "MASTERCARD", "AMEX", "DISCOVER", "DEBIT"]
FOOTERS = [
    "THANK YOU FOR SHOPPING!", "PLEASE COME AGAIN", "HAVE A NICE DAY!",
    "THANK YOU!", "RETURNS WITHIN 30 DAYS WITH RECEIPT",
    "SAVE YOUR RECEIPT", "TELL US HOW WE DID AT SURVEY.EXAMPLE.COM",
]

WIDTH = 40  # characters per line, typical of 80mm thermal printers


# --------------------------------------------------------------------------
# Clean receipt generation
# --------------------------------------------------------------------------

def money(x):
    return f"{x:,.2f}"


def center(text):
    return text.center(WIDTH).rstrip()


def row(left, right):
    space = max(1, WIDTH - len(left) - len(right))
    return f"{left}{' ' * space}{right}"


def random_date(rng):
    start = datetime(2020, 1, 1)
    dt = start + timedelta(minutes=rng.randint(0, 6 * 365 * 24 * 60))
    fmt = rng.choice([
        "%m/%d/%Y %I:%M %p", "%m/%d/%y %H:%M", "%Y-%m-%d %H:%M:%S",
        "%d-%b-%Y %H:%M", "%b %d, %Y %I:%M%p",
    ])
    return dt.strftime(fmt).upper()


def generate_receipt(rng):
    """Return a clean receipt as a list of lines plus structured metadata."""
    category = rng.choice(list(STORES))
    store = rng.choice(STORES[category])
    city, state = rng.choice(CITIES)
    lines = [
        center(store),
        center(f"{rng.randint(10, 9999)} {rng.choice(STREETS)}"),
        center(f"{city}, {state} {rng.randint(10000, 99999)}"),
        center(f"TEL ({rng.randint(200, 989)}) {rng.randint(200, 999)}-"
               f"{rng.randint(0, 9999):04d}"),
        "",
        row(random_date(rng), f"#{rng.randint(1000, 999999)}"),
        row(f"CASHIER: {rng.choice(CASHIERS)}", f"REG {rng.randint(1, 12)}"),
    ]
    if category == "restaurant":
        lines.append(row(f"TABLE {rng.randint(1, 40)}",
                         f"GUESTS {rng.randint(1, 6)}"))
    lines.append("-" * WIDTH)

    subtotal = 0.0
    pool = ITEMS[category]
    for name, lo, hi in rng.sample(pool, rng.randint(1, min(10, len(pool)))):
        unit = round(rng.uniform(lo, hi), 2)
        qty = rng.choices([1, 2, 3, 4], weights=[70, 18, 8, 4])[0]
        total = round(unit * qty, 2)
        subtotal += total
        if qty > 1:
            lines.append(name)
            lines.append(row(f"  {qty} @ {money(unit)}", money(total)))
        else:
            lines.append(row(name, money(total)))
        if rng.random() < 0.08:
            disc = round(total * rng.choice([0.1, 0.15, 0.2]), 2)
            subtotal -= disc
            lines.append(row("  DISCOUNT", f"-{money(disc)}"))

    subtotal = round(subtotal, 2)
    tax_rate = rng.choice([0.0, 0.05, 0.0625, 0.07, 0.0825, 0.0875, 0.1])
    tax = round(subtotal * tax_rate, 2)
    tip = 0.0
    lines.append("-" * WIDTH)
    lines.append(row("SUBTOTAL", money(subtotal)))
    if tax_rate:
        lines.append(row(f"TAX {tax_rate * 100:g}%", money(tax)))
    if category == "restaurant" and rng.random() < 0.6:
        tip = round(subtotal * rng.choice([0.15, 0.18, 0.2, 0.22]), 2)
        lines.append(row("TIP", money(tip)))
    total = round(subtotal + tax + tip, 2)
    lines.append(row("TOTAL", money(total)))
    lines.append("")

    if rng.random() < 0.25:
        tendered = float(int(total) + rng.choice([1, 5, 10, 20]))
        lines.append(row("CASH", money(tendered)))
        lines.append(row("CHANGE", money(tendered - total)))
        payment = "CASH"
    else:
        payment = rng.choice(CARDS)
        lines.append(row(f"{payment} ****{rng.randint(0, 9999):04d}",
                         money(total)))
        lines.append(f"AUTH CODE: {rng.randint(0, 999999):06d}")

    lines.append("")
    lines.append(center(rng.choice(FOOTERS)))
    if rng.random() < 0.5:
        lines.append(center(rng.choice(FOOTERS)))

    meta = {"category": category, "store": store, "total": money(total),
            "payment": payment}
    return lines, meta


# --------------------------------------------------------------------------
# OCR noise model
# --------------------------------------------------------------------------

# Visually confusable characters common in OCR output on receipts.
CONFUSIONS = {
    "0": ["O", "o", "D", "Q"], "O": ["0", "Q", "D"], "o": ["0", "c"],
    "1": ["l", "I", "i", "|", "7"], "l": ["1", "I", "|"], "I": ["1", "l", "|"],
    "2": ["Z", "z"], "Z": ["2"], "5": ["S", "s"], "S": ["5", "$"],
    "6": ["G", "b"], "G": ["6", "C"], "8": ["B", "3"], "B": ["8", "3"],
    "9": ["g", "q"], "4": ["A"], "A": ["4"], "7": ["1", "T"], "3": ["8"],
    "E": ["F", "3"], "F": ["E", "P"], "C": ["G", "("], "D": ["0", "O"],
    "U": ["V", "LI"], "V": ["U", "Y"], "H": ["N", "#"], "N": ["H", "M"],
    "M": ["N", "RN"], "W": ["VV"], "T": ["7", "I"], "R": ["P", "K"],
    "P": ["R", "F"], "K": ["X"], "Q": ["O", "0"],
    ".": [",", ":", ""], ",": [".", ""], ":": [";", "."], "-": ["_", "~", ""],
    "$": ["S", "5"], "/": ["l", "|"], "%": ["X"], "#": ["H"],
}
MULTI_CONFUSIONS = [("RN", "M"), ("M", "RN"), ("CL", "D"), ("VV", "W"),
                    ("W", "VV"), ("LI", "U"), ("NN", "M")]
JUNK = list(".,'`:;~-_|*")

LEVELS = {  # per-character probabilities for each noise operation
    "light":  dict(sub=0.015, drop=0.005, dup=0.003, ins=0.002, space=0.01,
                   case=0.002, line_drop=0.0, line_merge=0.0),
    "medium": dict(sub=0.04, drop=0.012, dup=0.006, ins=0.005, space=0.025,
                   case=0.006, line_drop=0.01, line_merge=0.02),
    "heavy":  dict(sub=0.08, drop=0.025, dup=0.012, ins=0.012, space=0.05,
                   case=0.015, line_drop=0.02, line_merge=0.05),
}


def corrupt_line(line, rng, p):
    # Thermal printers often lose column alignment in OCR: collapse runs of
    # spaces between the item name and price.
    if rng.random() < 0.6:
        parts = line.split()
        if parts and line.strip():
            gap = " " * rng.choice([1, 1, 2, 3])
            lead = line[:len(line) - len(line.lstrip())] if rng.random() < 0.3 else ""
            line = lead + gap.join(parts)

    s = line.upper()
    for a, b in MULTI_CONFUSIONS:
        if a in s and rng.random() < p["sub"] * 3:
            idx = s.index(a)
            line = line[:idx] + b + line[idx + len(a):]
            s = line.upper()

    out = []
    for ch in line:
        r = rng.random()
        if ch == " ":
            if r < p["space"]:
                continue  # merged words
            if r < p["space"] * 2:
                out.append("  ")
                continue
            out.append(ch)
            continue
        if r < p["drop"]:
            continue
        r -= p["drop"]
        if r < p["sub"] and ch in CONFUSIONS:
            out.append(rng.choice(CONFUSIONS[ch]))
            continue
        r -= p["sub"]
        if r < p["dup"]:
            out.append(ch * 2)
            continue
        r -= p["dup"]
        if r < p["case"] and ch.isalpha():
            out.append(ch.swapcase())
            continue
        out.append(ch)
        if rng.random() < p["ins"]:
            out.append(rng.choice(JUNK))
        elif ch.isalpha() and rng.random() < p["space"] / 3:
            out.append(" ")  # split word
    return "".join(out)


def corrupt_receipt(lines, rng, level):
    p = LEVELS[level]
    noisy = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() and rng.random() < p["line_drop"]:
            i += 1
            continue
        if i + 1 < len(lines) and line.strip() and rng.random() < p["line_merge"]:
            line = line + " " + lines[i + 1].strip()
            i += 1
        noisy.append(corrupt_line(line, rng, p))
        i += 1
    # OCR frequently drops blank lines and smears separator rows.
    noisy = [l for l in noisy if l.strip() or rng.random() < 0.5]
    return noisy


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

def make_examples(n, rng, granularity, level_weights, prefix):
    levels = list(level_weights)
    weights = list(level_weights.values())
    produced = 0
    rid = 0
    while produced < n:
        clean_lines, meta = generate_receipt(rng)
        level = rng.choices(levels, weights=weights)[0]
        if granularity == "receipt":
            noisy_lines = corrupt_receipt(clean_lines, rng, level)
            yield {
                "id": f"{prefix}-{rid:06d}",
                "input": "\n".join(noisy_lines),
                "target": "\n".join(l.rstrip() for l in clean_lines),
                "noise_level": level,
                **meta,
            }
            produced += 1
        else:
            for j, line in enumerate(clean_lines):
                if not line.strip() or produced >= n:
                    continue
                yield {
                    "id": f"{prefix}-{rid:06d}-l{j:02d}",
                    "input": corrupt_line(line, rng, LEVELS[level]),
                    "target": " ".join(line.split()),
                    "noise_level": level,
                    "category": meta["category"],
                }
                produced += 1
        rid += 1


def write_jsonl(path, rows):
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row_ in rows:
            f.write(json.dumps(row_, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--granularity", choices=["receipt", "line"],
                    default="receipt",
                    help="one example per full receipt, or per line")
    ap.add_argument("--levels", default="light:0.3,medium:0.5,heavy:0.2",
                    help="noise level mix, e.g. light:1 for light only")
    ap.add_argument("--clean-ratio", type=float, default=0.05,
                    help="fraction of examples left uncorrupted, so the "
                         "model learns not to change correct text")
    args = ap.parse_args()

    level_weights = {}
    for part in args.levels.split(","):
        name, w = part.split(":")
        if name not in LEVELS:
            ap.error(f"unknown noise level {name!r}; choose from {list(LEVELS)}")
        level_weights[name] = float(w)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, n in [("train", args.train), ("val", args.val), ("test", args.test)]:
        # Separate RNG per split keeps splits stable when sizes change.
        rng = random.Random(f"{args.seed}-{split}")
        rows = []
        for ex in make_examples(n, rng, args.granularity, level_weights, split):
            if rng.random() < args.clean_ratio:
                ex["input"] = ex["target"]
                ex["noise_level"] = "none"
            rows.append(ex)
        count = write_jsonl(out / f"{split}.jsonl", rows)
        print(f"wrote {count:>6} examples -> {out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
