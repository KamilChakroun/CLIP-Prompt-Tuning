"""Train CoOp prompts for CLIP fire vs. not-fire classification.

CLIP stays frozen. Only the context token embeddings are learned (see clip_fire.py).
Image features are computed once and cached, so each epoch takes seconds.

Usage:
  python train_clip_fire.py --fire DIR --not-fire DIR [--out learned_prompts_coop.pth]
  (or set TRAIN_FIRE_PATH / TRAIN_NONFIRE_PATH in .env or the shell)

The checkpoint keeps the epoch with the lowest validation loss, plus the P(fire) threshold that
keeps --target-recall of the validation fire images.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from clip_fire import (FIRE, PromptLearner, default_quick_gelu, encode_images, list_images, load_clip, load_env, operating_point,
                       pick_device, save_checkpoint, seed_everything, threshold_at_recall)


def parse_args():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fire", default=os.getenv("TRAIN_FIRE_PATH"), help="folder of fire images (recursive)")
    ap.add_argument("--not-fire", default=os.getenv("TRAIN_NONFIRE_PATH"), help="folder of non-fire images")
    ap.add_argument("--out", default="learned_prompts_coop.pth")
    ap.add_argument("--class-names", nargs=2, default=["fire", "normal scene"],
                    help="text for class 0 (fire) and class 1. Avoid negations like 'non fire'.")
    ap.add_argument("--arch", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--resize-mode", default="longest", choices=["longest", "shortest", "squash"])
    ap.add_argument("--n-ctx", type=int, default=16)
    ap.add_argument("--ctx-init", default="", help='e.g. "a photo of a"; empty = random init')
    ap.add_argument("--csc", action="store_true", help="class-specific context")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--target-recall", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--workers", type=int, default=0)
    return ap.parse_args()


def evaluate(model, learner, feats, y, class_w):
    with torch.no_grad():
        w = learner.class_weights(model)
        logits = model.logit_scale.exp().float() * feats @ w.T
        loss = F.cross_entropy(logits, y, weight=class_w).item()
        p = logits.softmax(-1)[:, FIRE].cpu().numpy()
    is_fire = (y == FIRE).cpu().numpy()
    acc = float(((p >= 0.5) == is_fire).mean())
    auc = roc_auc_score(is_fire, p) if 0 < is_fire.sum() < len(is_fire) else float("nan")
    return loss, acc, auc, p, is_fire


def main():
    a = parse_args()
    seed_everything(a.seed)
    device = pick_device(a.device)
    fire, other = list_images(a.fire), list_images(a.not_fire)
    if not fire or not other:
        raise SystemExit(f"need images in both folders: fire={len(fire)} not-fire={len(other)}")
    paths = fire + other
    labels = np.array([0] * len(fire) + [1] * len(other))
    tr, va = train_test_split(np.arange(len(paths)), test_size=a.val_split, stratify=labels, random_state=a.seed)
    print(f"images: fire={len(fire)} not-fire={len(other)} | train={len(tr)} val={len(va)} (stratified, seed {a.seed})")
    print("note: this split is per image. Frames from one video can land in both train and val, "
          "so val is optimistic. Prefer a test set from different videos/cameras.")

    model, preprocess, tokenizer = load_clip(a.arch, a.pretrained, a.resize_mode, device)
    feats, ok = encode_images(model, preprocess, paths, device, workers=a.workers)
    tr, va = tr[ok[tr].numpy()], va[ok[va].numpy()]
    feats, y = feats.to(device), torch.as_tensor(labels, device=device)

    counts = np.bincount(labels[tr], minlength=2)
    class_w = torch.tensor(len(tr) / (2 * counts), dtype=torch.float32, device=device)  # balanced
    learner = PromptLearner(model, tokenizer, a.class_names, a.n_ctx, a.ctx_init, a.csc)
    opt = torch.optim.AdamW([learner.ctx], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    g = torch.Generator().manual_seed(a.seed)
    scale = model.logit_scale.exp().float()

    best = {"val_loss": float("inf")}
    for ep in range(1, a.epochs + 1):
        learner.train()
        perm = torch.as_tensor(tr)[torch.randperm(len(tr), generator=g)]
        tot = 0.0
        for i in range(0, len(perm), a.batch_size):
            idx = perm[i:i + a.batch_size].to(device)
            logits = scale * feats[idx] @ learner.class_weights(model).T
            loss = F.cross_entropy(logits, y[idx], weight=class_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        learner.eval()
        vl, vacc, vauc, p, is_fire = evaluate(model, learner, feats[va], y[va], class_w)
        mark = ""
        if vl < best["val_loss"]:
            t = threshold_at_recall(p, is_fire, a.target_recall)
            best = {"val_loss": vl, "epoch": ep, "val_acc": vacc, "val_auc": vauc,
                    "op": operating_point(p, is_fire, t), "ctx": learner.ctx.detach().clone()}
            mark = "  *best"
        print(f"epoch {ep:3d} | train loss {tot / len(tr):.4f} | val loss {vl:.4f} acc {vacc:.4f} auc {vauc:.4f}{mark}")

    learner.ctx.data.copy_(best.pop("ctx"))
    op = best["op"]
    save_checkpoint(a.out, learner, {
        "arch": a.arch, "pretrained": a.pretrained, "resize_mode": a.resize_mode,
        "quick_gelu": default_quick_gelu(a.pretrained), "threshold": op["threshold"], "target_recall": a.target_recall, "val": best, "args": vars(a),
        "n_train": {"fire": int(counts[0]), "not_fire": int(counts[1])}})
    print(f"\nbest epoch {best['epoch']}: val loss {best['val_loss']:.4f} acc {best['val_acc']:.4f} auc {best['val_auc']:.4f}")
    print(f"threshold for >= {a.target_recall:.0%} fire recall on val: P(fire) >= {op['threshold']:.4f} "
          f"-> fire recall {op['fire_recall']:.4f}, non-fire rejected {op['false_alarms_rejected']:.4f}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
