#!/usr/bin/env python3
"""Baseline classifier and evaluator for the item-name dataset.

Trains on train.jsonl using character n-gram TF-IDF features, then reports
accuracy on test.jsonl broken down by OCR noise level, category and unseen
brand. Use it as the score your real model has to beat.

    pip install scikit-learn
    python baseline.py --data data_items
    python baseline.py --data data_uk --test real_test/sainsburys_2026-10-05.jsonl

--model nn (default) predicts the target of the most similar training line,
so it works for thousands of item names. --model logreg trains a logistic
regression classifier: slower, and only sensible for --target generic.

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
        keys = ["overall", f"noise={r.get('ocr_noise')}", f"category={r.get('category')}"]
        if r["target"] == "Non-Item" or normalise(predictions[r["id"]]) == "non-item":
            keys.append("non_item_lines" if r["target"] == "Non-Item" else "item_lines")
        else:
            keys.append("item_lines")
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
    ap.add_argument("--test", help="test file (default: <data>/test.jsonl); use a "
                                   "hand-labelled real_test file to measure real accuracy")
    ap.add_argument("--model", choices=["nn", "logreg"], default="nn")
    ap.add_argument("--predictions", help="score this file instead of training")
    ap.add_argument("--errors", type=int, default=15, help="mistakes to print")
    args = ap.parse_args()

    data = Path(args.data)
    test = load(args.test or data / "test.jsonl")

    if args.predictions:
        predictions = {p["id"]: p["prediction"] for p in load(args.predictions)}
        missing = [r["id"] for r in test if r["id"] not in predictions]
        if missing:
            raise SystemExit(f"{len(missing)} test ids have no prediction, e.g. {missing[:3]}")
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        train = load(data / "train.jsonl")
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=True,
                              sublinear_tf=True, min_df=2)
        print(f"training {args.model} on {len(train)} examples ...")
        x_train = vec.fit_transform(r["input"] for r in train)
        x_test = vec.transform(r["input"] for r in test)
        if args.model == "nn":
            labels = [r["target"] for r in train]
            guesses = []
            for start in range(0, x_test.shape[0], 500):  # keep memory bounded
                sims = x_test[start:start + 500] @ x_train.T
                guesses += [labels[i] for i in sims.argmax(axis=1).A1]
        else:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, C=20)
            clf.fit(x_train, [r["target"] for r in train])
            guesses = map(str, clf.predict(x_test))
        predictions = dict(zip((r["id"] for r in test), guesses))

    report(test, predictions)
    wrong = [r for r in test if normalise(predictions[r["id"]]) != normalise(r["target"])]
    if wrong and args.errors:
        print(f"\nsample errors ({len(wrong)} total):")
        for r in wrong[:args.errors]:
            print(f"  {r['input']:<44} want {r['target']!r:<24} got {predictions[r['id']]!r}")


if __name__ == "__main__":
    main()
