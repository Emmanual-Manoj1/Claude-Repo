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
