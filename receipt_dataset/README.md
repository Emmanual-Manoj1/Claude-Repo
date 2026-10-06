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

# Item-Name Dataset

`generate_items.py` turns one OCR'd receipt line into the **item name only**,
with no price, size or quantity. Lines that aren't purchased items become
`Non-Item`.

```
MLIGHT STRAWBRY 160G £O 40        -> Muller Light Strawberry
TUNA 8 SWEETCORN SW 2 05          -> Tuna & Sweetcorn Sandwich
*RIBENA BLK 50OML 160             -> Ribena Blackcurrant
DORIT0S COOU43G O 90              -> Doritos Cool Original
2 X Meal Deal 155                 -> Meal Deal
13 Sub Total- 2245                -> Non-Item
PROMOT IONS O1.55                 -> Non-Item
Vat NuMbe A 6604548 36            -> Non-Item
```

Every row also carries a `generic` product type (Yogurt, Sandwich, Fruit
Drink). Train on that instead with `--target generic`.

## Quick start (UK receipt scanner)

```bash
python generate_items.py --locale uk --out data_uk     # 50k / 5k / 5k
pip install scikit-learn
python baseline.py --data data_uk --test real_test/sainsburys_2026-10-05.jsonl
```

Use `--locale us` (the default) for US receipts.

## Built from real scans

`real_logs/` holds ReceiptOCR logcat output from real Sainsbury's scans. The
UK settings are tuned to what that OCR actually does:

| What the OCR did | Example | How the generator copies it |
|---|---|---|
| Misread `£` as a letter or digit | `E450`, `C0.80`, `L22.20`, `e1.30`, `A1 50`, `P22.45`, `9160`, `22.05` | `POUND_MISREADS` |
| Dropped or spaced the decimal point | `450`, `1 55`, `£O 90` | `uk_amount()` |
| Misread `*` markers | `*RIBENA`, `xMONSTER`, `*\|ICTAC` | UK `fillers_pre` |
| Printed brands in short form | `MLIGHT` (Muller Light), `TICTAC`, `JS` (Sainsbury's) | brand forms in `catalogue_uk.txt` |
| Merged or split words | `BLK500ML`, `MUIGHTSTRAWBRY`, `PROMOT IONS` | OCR noise model |
| Printed many header and total lines | address, phone, VAT number, `13 Sub Total`, `PROMOTIONS` | 30% `Non-Item` lines |

[`real_test/sainsburys_2026-10-05.jsonl`](real_test/) is those scans
hand-labelled: 78 unique rows, 42 items and 36 non-items. **This is the
number to watch.** It shows how a model does on your real scanner, not on
synthetic data.

To label more of your own scans:

```bash
python parse_logs.py my_scans/*.log --out to_label.jsonl   # then fill in "target"
python baseline.py --data data_uk --test to_label.jsonl
```

`parse_logs.py` removes duplicate rows from repeat scans. It also skips scans
where the OCR merged whole columns into a few rows (for example, all prices
on one line), because no line-level model can fix those. See "Scanner
pipeline" below.

## Baseline results

`baseline.py --model nn` predicts the item name of the most similar training
line, using character 1–4-gram TF-IDF features.

| Test set | Overall | Items | Non-items |
|---|---|---|---|
| **Real Sainsbury's scans** (UK data, 50k train) | **88.5%** | 78.6% | 100% |
| Synthetic UK test | 78.1% | 74.3% | 89.2% |
| Synthetic US test, `--target generic` | 94.5% | 95.3% | 81.2% |

The remaining real-scan errors are near-misses, such as
`CHICKEN NOODLES → Supernoodles Chicken Noodles` and
`MONSTER MUNCH → Monster Munch Crisps`. A nearest-neighbour model can only
copy names it has seen, so it also does badly on brands missing from
training (9% on held-out brands). A **generative model** such as ByT5/T5 or a
fine-tuned LLM using the `*_chat.jsonl` files should do clearly better on
both counts.

## Scanner pipeline

On a phone (for example ML Kit text recognition):

1. **Rebuild rows from bounding boxes** before anything else. Two of the
   five original scans came back as column blocks (one is kept in
   `real_logs/`): all prices in one row, all
   names in another. Group OCR lines whose vertical centres are within about
   half a line height, then sort each group left to right. A text model
   can't recover from this, so it has to be fixed in the app.
2. **Run the model on each row** to get the item name, or `Non-Item`.
3. **Keep reading after the first total.** In the logged scans, `TICTAC FRUIT AD`
   comes after the first `TOTAL TO PAY`, so a parser that stops there misses
   it. With a model that labels every row, you don't need a fixed "item
   block".

## Output files

| file | use |
|---|---|
| `train/val/test.jsonl` | main format, with all metadata |
| `train/val/test_chat.jsonl` | `{"messages": [system, user, assistant]}` for LLM fine-tuning |
| `train/val/test.csv` | spreadsheets, pandas, AutoML tools |
| `labels.txt` | every target that appears |
| `stats.json` | per-split counts by category, noise level and held-out slice |

A JSONL row:

```json
{"id": "train-000012", "input": "*RIBENA BLK 500ML 1 60", "target": "Ribena Blackcurrant",
 "name": "Ribena Blackcurrant", "generic": "Fruit Drink", "category": "drinks",
 "brand": "RIBENA", "size": "500ML", "ocr_noise": "none",
 "unseen_brand": false, "unseen_product": false}
```

Committed samples (1000 / 200 / 200): [`sample_items/`](sample_items/) (US)
and [`sample_items_uk/`](sample_items_uk/) (UK).

## Options

```bash
--locale uk|us            # receipt style and catalogue
--target name|generic     # item name (default) or product type
--train 200000            # more data; 60k takes about 3 seconds
--holdout-brands 0.15     # some brands appear only in val/test
--holdout-products 0.1    # some product groups appear only in val/test
--noise none:1            # abbreviations only, no OCR noise
--non-item-ratio 0.3      # share of header/total/promo lines
--price-ratio 0.85        # share of item lines with a price
--formats jsonl,chat,csv
```

## Catalogues

[`catalogue_uk.txt`](catalogue_uk.txt) has about 730 item names in 115 product
groups, with about 340 brands. [`catalogue.txt`](catalogue.txt) (US) has 320
products. One product group per line:

```
generic | category | brands | sizes | item names
Yogurt|dairy|Muller Light=MLIGHT/MULLER LIGHT,ACTIVIA|160G,4X120G|STRAWBERRY=STRAWBRY/STRAWB;VANILLA
```

- `Clean Name=PRINTED/OTHER PRINTED` says how a brand or item is printed on
  receipts. The clean part becomes the target, so `COOL ORIGINAL=COOL` teaches
  `DORITOS COOL → Doritos Cool Original`. Mixed-case clean names are kept
  exactly as written (`CBeebies Weekly`).
- Brand `-` means unbranded or store brand, so a store code like `JS` or
  `TESCO` may be printed in front and dropped from the name. `NONE` means the
  item is never branded (magazines, meal deals).
- The loader rejects malformed lines and any item name listed under two
  product groups.

The fastest way to raise real-scan accuracy is to add the products you
actually buy, using the forms your receipts print them in, and to label a
few more real scans.
