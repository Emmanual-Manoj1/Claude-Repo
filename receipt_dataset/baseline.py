#!/usr/bin/env python3
"""Baseline classifier and evaluator for the item-name dataset.

Trains character n-gram TF-IDF + logistic regression on train.jsonl and
reports accuracy on test.jsonl, broken down by OCR noise level, category and
unseen brand. Use it as the score your real model has to beat.

    pip install scikit-learn
    python baseline.py --data data_items

To score your own model, write one JSON line per test example with
{"id": ..., "prediction": ...} and run:

    python baseline.py --data data_items --predictions my_preds.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalise(label):
    return " ".join(label.lower().split())


def report(rows, predictions):
    buckets = defaultdict(lambda: [0, 0])
    for r in rows:
        ok = normalise(predictions[r["id"]]) == normalise(r["target"])
        keys = ["overall", f"noise={r['ocr_noise']}", f"category={r['category']}"]
        if r.get("unseen_brand"):
            keys.append("unseen_brand")
        if r.get("unseen_product"):
            keys.append("unseen_product")
        for k in keys:
            buckets[k][0] += ok
            buckets[k][1] += 1
    order = sorted(buckets, key=lambda k: (k != "overall", k.split("=")[0], k))
    for k in order:
        right, total = buckets[k]
        print(f"{k:<28} {right / total:7.2%}  ({right}/{total})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_items")
    ap.add_argument("--predictions", help="score this file instead of training")
    ap.add_argument("--errors", type=int, default=15, help="mistakes to print")
    args = ap.parse_args()

    data = Path(args.data)
    test = load(data / "test.jsonl")

    if args.predictions:
        predictions = {p["id"]: p["prediction"] for p in load(args.predictions)}
        missing = [r["id"] for r in test if r["id"] not in predictions]
        if missing:
            raise SystemExit(f"{len(missing)} test ids have no prediction, e.g. {missing[:3]}")
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        train = load(data / "train.jsonl")
        model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True,
                            sublinear_tf=True, min_df=2),
            LogisticRegression(max_iter=1000, C=20),
        )
        print(f"training on {len(train)} examples ...")
        model.fit([r["input"] for r in train], [r["target"] for r in train])
        predictions = dict(zip((r["id"] for r in test), map(str,
                               model.predict([r["input"] for r in test]))))

    report(test, predictions)
    wrong = [r for r in test if normalise(predictions[r["id"]]) != normalise(r["target"])]
    if wrong and args.errors:
        print(f"\nsample errors ({len(wrong)} total):")
        for r in wrong[:args.errors]:
            print(f"  {r['input']:<44} want {r['target']!r:<24} got {predictions[r['id']]!r}")


if __name__ == "__main__":
    main()
