# Receipt OCR Text-Correction Dataset

A synthetic dataset for training a model that fixes OCR errors in receipt
text. Each example pairs a **noisy OCR-style input** with the **clean target**
receipt.

## Quick start

```bash
# Full dataset (8k / 1k / 1k receipts) -> data/  (git-ignored)
python generate.py

# Bigger, line-level, light noise only
python generate.py --train 50000 --granularity line --levels light:1
```

Requires only Python 3.8+ (no dependencies). Output is deterministic for a
given `--seed`.

A small committed sample (200 / 50 / 50) lives in [`sample/`](sample/).

## Format

JSONL, one example per line:

```json
{
  "id": "train-000042",
  "input": "FRESH MART\n...SUBT0TAL  12.4O\n...",
  "target": "              FRESH MART\n...SUBTOTAL                           12.40\n...",
  "noise_level": "medium",
  "category": "grocery",
  "store": "FRESH MART",
  "total": "12.40",
  "payment": "VISA"
}
```

| field | meaning |
|---|---|
| `input` | corrupted text (model input) |
| `target` | correct text (model output) |
| `noise_level` | `none`, `light`, `medium` or `heavy` |
| `category` | grocery, restaurant, coffee, pharmacy, hardware, gas |
| `store`, `total`, `payment` | ground-truth fields (receipt mode only), handy for extraction evals |

With `--granularity line`, each example is one receipt line and whitespace in
the target is collapsed to single spaces.

## What the noise model simulates

- **Character confusions:** `0↔O`, `1↔l↔I`, `5↔S`, `8↔B`, `C→G`, `P→R`, `$→S`, etc.
- **Multi-character confusions:** `rn↔m`, `cl→d`, `vv↔w`
- **Dropped, duplicated and inserted characters**, including stray `.,'|~` specks
- **Spacing:** merged words, split words, lost column alignment between item and price
- **Line-level errors:** dropped lines, merged lines, missing blank lines
- **Decimal damage:** `12.99 → 12,99 / 12:99 / 1299`
- **Clean passthroughs:** about 5% of examples are left uncorrupted (`--clean-ratio`), so the model learns not to "fix" text that is already correct

Noise mix is set with `--levels`, default `light:0.3,medium:0.5,heavy:0.2`.

## Using it for training

Typical seq2seq setup (T5/ByT5/BART) or chat fine-tuning:

```python
import json
rows = [json.loads(l) for l in open("data/train.jsonl")]
pairs = [("fix receipt: " + r["input"], r["target"]) for r in rows]
```

Character-level models such as ByT5 tend to work well for OCR correction.
For evaluation, report character error rate (CER) on `test.jsonl`, broken
down by `noise_level`.

## Tips

- If you have **real** receipt OCR output, mix some in. Even a few hundred
  hand-corrected real examples help a lot, because synthetic noise never
  matches a real OCR engine exactly.
- To match your engine's habits, tune `CONFUSIONS` and `LEVELS` in
  `generate.py`.

---

# Item-Name Normalisation Dataset

`generate_items.py` maps raw receipt line items to **generic product names**:

```
LINDT DARK CHOC BR 80 G FT                 -> Dark Chocolate
Pure Premium Orange Juice 52Oz Tf          -> Orange Juice
T BLUE BUFFALO ADUL DOG FOOD CHKN   2.33 N -> Dog Food
VALU GROUND  CHUCK 1LB FF  3.29            -> Ground Beef
ACT MTH RINS 1L                            -> Mouthwash
RAGU   TRAD|TIONAL  PA5Ta    $AUCE   24.08 -> Pasta Sauce
Return Sp Bars -1.90                       -> Non-Item
PAPER BAG   28',10                         -> Non-Item
```

The model learns to drop the brand, size, pack count, PLU code, tax flags and
price, to expand POS abbreviations, to see through OCR errors, and to
recognise lines that are not products.

```bash
python generate_items.py                          # 50k / 5k / 5k -> data_items/
python generate_items.py --train 200000           # more data
python generate_items.py --holdout-brands 0.15    # some brands only in val/test
python generate_items.py --holdout-products 0.1   # some products only in val/test
python generate_items.py --noise none:1           # abbreviations only, no OCR noise
```

Generating 60k examples takes about 3 seconds. A committed sample
(1000 / 200 / 200) is in [`sample_items/`](sample_items/).

## What's covered

- **320 products in 20 categories**: grocery, dairy, produce, meat, seafood,
  bakery, deli, frozen, snacks, beverages, alcohol, household, personal care,
  baby, pharmacy, pet, floral, automotive and general merchandise. There are
  about 550 real brands and about 1,200 printed name variants.
- **About 360 curated POS abbreviations** (`CHOCOLATE→CHOC/CHC`,
  `BONELESS→BNLS`, `DISHWASHER→DW`), plus automatic vowel-dropping and
  truncation for everything else.
- **Brand variants:** full, first word or initials, plus store-brand codes
  (`GV`, `KS`, `365`).
- **Layout:** brand, name and size in any order; PLU codes; tax flags;
  fixed-width truncation (never into the product name); and in 30% of lines,
  prices, `2 @ 1.99` multi-buys and `1.32 LB @ 0.59/LB` weights.
- **Casing:** mostly upper case, with some Title Case and lower case lines.
- **Non-items (6%):** totals, tax, payment, deposits, bag fees, savings,
  reference numbers. The hard negatives are coupons and returns that *name a
  product* (`COUPON LINDT CHOC -1.00`). All of these are labelled `Non-Item`.
- **OCR noise:** the same noise model as `generate.py`, in a
  `none/light/medium/heavy` mix.

Inputs are de-duplicated across splits, so no test input appears in training.

## Output files

| file | use |
|---|---|
| `train/val/test.jsonl` | main format, with all metadata |
| `train/val/test_chat.jsonl` | `{"messages": [system, user, assistant]}` for LLM fine-tuning |
| `train/val/test.csv` | spreadsheets, pandas, AutoML tools |
| `labels.txt` | the 321 target labels (320 products + `Non-Item`) |
| `stats.json` | per-split counts by category, noise level and held-out slice |

Turn formats off with `--formats jsonl`. A JSONL row:

```json
{"id": "train-000000", "input": "POWERADE 8 PK THIRST QUENC TF",
 "target": "Sports Drink", "category": "beverages", "brand": "POWERADE",
 "size": "8PK", "ocr_noise": "none", "unseen_brand": false, "unseen_product": false}
```

`brand` and `size` are ground truth, so the same data can also train an
extractor.

## Picking a model

- **Classifier** (fast and cheap): predict one of the 321 labels. See the
  baseline below.
- **Small seq2seq model** (T5 / ByT5 / BART): generates the name, so it can
  handle products outside the label list. Measure this with
  `--holdout-products`.
- **LLM fine-tune:** use the `*_chat.jsonl` files directly.

## Baseline and evaluation

`baseline.py` trains a character n-gram TF-IDF + logistic regression model,
then reports accuracy by noise level, category and held-out slice:

```bash
pip install scikit-learn
python baseline.py --data data_items
```

On the default dataset with `--holdout-brands 0.1` (50k train / 5k test):

| slice | accuracy |
|---|---|
| overall | 98.7% |
| no OCR noise | 99.9% |
| light / medium / heavy noise | 98.1% / 96.7% / 91.9% |
| unseen brands | 94.4% |

The remaining errors are the genuinely hard cases: `POTATO ROLLS → Potatoes`,
`CANADA DRY GING ALE → Ginger`, `HONEY BBQ SAUCE → Honey`. A good model
should beat this, especially on heavy noise and unseen brands.

To score your own model, write `{"id": ..., "prediction": ...}` lines for
`test.jsonl` and run `python baseline.py --data data_items --predictions preds.jsonl`.

> Synthetic accuracy overstates real-world accuracy. Before you trust the
> number, hand-label 200 or more real receipt lines from your own stores and
> evaluate on those.

## Extending

Products live in [`catalogue.txt`](catalogue.txt), one per line:

```
target | category | brands | sizes | name variants
Dark Chocolate|confectionery|LINDT,GHIRARDELLI|100G,3.5OZ|DARK CHOCOLATE;CHOCOLATE DARK 70%
```

The loader rejects malformed lines, duplicate targets, and any name variant
that is shared by two products. To add new abbreviations, edit `ABBREV` in
`generate_items.py`. You can point the generator at your own file with
`--catalogue my_products.txt`.
