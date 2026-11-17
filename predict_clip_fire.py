"""Predict P(fire) with a trained checkpoint and, if labels are known, evaluate it.

Usage:
  # labelled test set -> CSV + report
  python predict_clip_fire.py --fire DIR --not-fire DIR [--ckpt learned_prompts_coop.pth]
  # unlabelled folder -> CSV only
  python predict_clip_fire.py --images DIR
  (TEST_FIRE_PATH / TEST_NONFIRE_PATH from .env or the shell are used when no folder is given)

Checkpoints: new CoOp checkpoints from train_clip_fire.py, or the original learned_prompts.pth
(read with the formula it was trained with, CLIP center-crop preprocessing, threshold 0.5).

The threshold is, in order: --threshold, the one stored in the checkpoint (chosen on val for the
target fire recall), or 0.5. The report also shows the threshold that would hit --target-recall
on THIS set; use it to understand the trade-off, not to tune on the test set.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score

from clip_fire import (classifier_weights, encode_images, fire_prob, list_images, load_clip, load_env,
                       operating_point, pick_device, read_checkpoint, threshold_at_recall)


def parse_args():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_ckpt = "learned_prompts_coop.pth" if Path("learned_prompts_coop.pth").exists() else "learned_prompts.pth"
    ap.add_argument("--ckpt", default=default_ckpt)
    ap.add_argument("--fire", default=os.getenv("TEST_FIRE_PATH"))
    ap.add_argument("--not-fire", default=os.getenv("TEST_NONFIRE_PATH"))
    ap.add_argument("--images", help="unlabelled folder (overrides --fire/--not-fire)")
    ap.add_argument("--out", default="fire_predictions.csv")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--target-recall", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device")
    ap.add_argument("--workers", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse_args()
    device = pick_device(a.device)
    ck = read_checkpoint(a.ckpt)
    model, preprocess, tokenizer = load_clip(ck["arch"], ck["pretrained"], ck["resize_mode"], device,
                                             ck.get("quick_gelu"))
    weights = classifier_weights(model, tokenizer, ck, device)
    t = a.threshold if a.threshold is not None else (ck.get("threshold") or 0.5)
    print(f"checkpoint {a.ckpt} ({ck['format']}), classes {ck['class_names']}, "
          f"preprocessing '{ck['resize_mode']}', threshold {t:.4f}")

    if a.images:
        paths, truth = list_images(a.images), None
    else:
        fire, other = list_images(a.fire), list_images(a.not_fire)
        paths, truth = fire + other, np.array([True] * len(fire) + [False] * len(other))
    feats, ok = encode_images(model, preprocess, paths, device, a.batch_size, a.workers)
    ok = ok.numpy()
    p = fire_prob(model, feats, weights).numpy()

    df = pd.DataFrame({"image_path": [str(x) for x in paths], "p_fire": p.round(5),
                       "pred": np.where(p >= t, "fire", "not_fire"), "readable": ok})
    if truth is not None:
        df.insert(1, "true", np.where(truth, "fire", "not_fire"))
    df.loc[~ok, "p_fire"] = np.nan
    df.loc[~ok, "pred"] = ""
    df.to_csv(a.out, index=False)
    print(f"{len(df)} images -> {a.out}" + (f"  ({(~ok).sum()} unreadable, excluded)" if (~ok).any() else ""))
    if truth is None:
        print(f"predicted fire: {(df.pred == 'fire').sum()} / {ok.sum()}")
        return

    p, truth = p[ok], truth[ok]
    pred = p >= t
    print("\n" + classification_report(truth, pred, labels=[True, False], target_names=["fire", "not_fire"],
                                       digits=4, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[False, True]).ravel()
    print(f"confusion: fire caught {tp}, fire missed {fn}, non-fire rejected {tn}, non-fire flagged {fp}")
    if 0 < truth.sum() < len(truth):
        print(f"ROC-AUC {roc_auc_score(truth, p):.4f} | average precision {average_precision_score(truth, p):.4f}")
    op = operating_point(p, truth, t)
    print(f"at threshold {t:.4f}: fire recall {op['fire_recall']:.4f}, non-fire rejected {op['false_alarms_rejected']:.4f}")
    t2 = threshold_at_recall(p, truth, a.target_recall)
    op2 = operating_point(p, truth, t2)
    print(f"for reference, {a.target_recall:.0%} fire recall on this set needs threshold {t2:.4f} "
          f"-> non-fire rejected {op2['false_alarms_rejected']:.4f}")


if __name__ == "__main__":
    main()
