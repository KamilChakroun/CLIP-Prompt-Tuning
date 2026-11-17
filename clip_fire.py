"""Shared code for the CLIP fire classifier: model loading, CoOp prompt learner, features, metrics.

Both train_clip_fire.py and predict_clip_fire.py import from here, so the classifier that is
trained is exactly the classifier that is used at prediction time.

Method: CoOp (Zhou et al., 2022). Learnable context *token embeddings* are placed before the class
name and the whole prompt runs through CLIP's frozen text encoder:
    [SOS] [V1] ... [Vn] <class name> . [EOS]  ->  text encoder  ->  class weight
The image encoder is frozen too, so image features are computed once and cached.
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.transformer import text_global_pool
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp", ".webp"}
FIRE = 0  # class index of "fire" in every checkpoint; the other class is "not fire"
CKPT_FORMAT = "coop-v2"


# ---------------------------------------------------------------- environment / data helpers
def load_env(path: str | Path = Path(__file__).with_name(".env")) -> None:
    """Minimal .env reader (KEY=VALUE lines). Never overrides variables already set in the shell."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: str | Path | None) -> list[Path]:
    """All images under `folder`, recursively, in a stable order."""
    if not folder:
        raise SystemExit("image folder not set (pass it as an argument or set it in .env)")
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"folder not found: {folder}")
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def pick_device(name: str | None = None) -> str:
    return name or ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- CLIP
def default_quick_gelu(pretrained: str) -> bool:
    # OpenAI's weights were trained with QuickGELU; open_clip's plain "ViT-B-32" config uses GELU
    # and only warns about the mismatch. The original 2025 checkpoint was trained WITHOUT this fix.
    return pretrained == "openai"


def load_clip(arch: str, pretrained: str, resize_mode: str, device: str, quick_gelu: bool | None = None):
    """Frozen CLIP + its preprocessing.

    resize_mode (open_clip `image_resize_mode`):
      longest   pad the image to a square, then resize -> nothing is cut off (default; best for crops)
      shortest  resize the short side, then center-crop -> CLIP's default, cuts the edges of wide images
      squash    resize to a square without keeping the aspect ratio
    """
    if quick_gelu is None:
        quick_gelu = default_quick_gelu(pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, image_resize_mode=resize_mode, force_quick_gelu=quick_gelu)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, open_clip.get_tokenizer(arch)


class ImageFiles(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = list(paths), preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return self.preprocess(im.convert("RGB")), True
        except Exception as e:  # unreadable file: reported by the caller, never silently dropped
            print(f"unreadable image: {self.paths[i]} ({e})")
            return torch.zeros(3, 224, 224), False


@torch.no_grad()
def encode_images(model, preprocess, paths, device, batch_size=64, workers=0, desc="images"):
    """L2-normalised image features (float32, CPU) and a mask of files that could be read."""
    dl = DataLoader(ImageFiles(paths, preprocess), batch_size=batch_size, num_workers=workers)
    feats, ok = [], []
    for x, good in tqdm(dl, desc=f"encoding {desc}"):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            f = model.encode_image(x.to(device))
        feats.append(F.normalize(f.float(), dim=-1).cpu())
        ok.append(good)
    return torch.cat(feats), torch.cat(ok)


def encode_token_embeddings(model, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """open_clip's CLIP.encode_text, starting from token embeddings instead of token ids."""
    cast = model.transformer.get_cast_dtype()
    x = x.to(cast) + model.positional_embedding.to(cast)
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = model.ln_final(x)
    x = text_global_pool(x, tokens, model.text_pool_type, eos_token_id=getattr(model, "text_eos_id", None))
    if model.text_projection is not None:
        x = model.text_projection(x) if isinstance(model.text_projection, nn.Linear) else x @ model.text_projection
    return F.normalize(x.float(), dim=-1)


# ---------------------------------------------------------------- CoOp prompt learner
class PromptLearner(nn.Module):
    """Learnable context vectors in CLIP's token-embedding space.

    n_ctx     number of context tokens (ignored when ctx_init is given: its token count is used)
    ctx_init  e.g. "a photo of a": start the context from these words instead of random noise
    csc       class-specific context (one context per class) instead of one shared context
    """

    def __init__(self, model, tokenizer, class_names, n_ctx=16, ctx_init="", csc=False):
        super().__init__()
        width = model.token_embedding.weight.shape[1]
        dev = model.token_embedding.weight.device
        if ctx_init:
            ids = tokenizer([ctx_init])[0]
            n_ctx = int((ids != 0).sum()) - 2  # minus [SOS] and [EOS]
            with torch.no_grad():
                init = model.token_embedding(ids.unsqueeze(0).to(dev))[0, 1:1 + n_ctx].float()
            ctx = init.unsqueeze(0).repeat(len(class_names), 1, 1) if csc else init
            prefix_words = ctx_init
        else:
            shape = (len(class_names), n_ctx, width) if csc else (n_ctx, width)
            ctx = torch.empty(shape, device=dev).normal_(std=0.02)
            prefix_words = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx.clone())
        self.n_ctx, self.csc, self.ctx_init, self.class_names = n_ctx, csc, ctx_init, list(class_names)
        tokens = tokenizer([f"{prefix_words} {c}." for c in class_names]).to(dev)
        with torch.no_grad():
            emb = model.token_embedding(tokens).float()
        self.register_buffer("tokens", tokens, persistent=False)
        self.register_buffer("prefix", emb[:, :1], persistent=False)          # [SOS]
        self.register_buffer("suffix", emb[:, 1 + n_ctx:], persistent=False)  # class name . [EOS] padding

    def forward(self) -> torch.Tensor:
        ctx = self.ctx if self.csc else self.ctx.unsqueeze(0).expand(len(self.class_names), -1, -1)
        return torch.cat([self.prefix, ctx, self.suffix], dim=1)

    def class_weights(self, model) -> torch.Tensor:
        """(n_classes, d) L2-normalised text features, differentiable w.r.t. self.ctx."""
        return encode_token_embeddings(model, self(), self.tokens)


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(path, learner: PromptLearner, meta: dict) -> None:
    torch.save({"format": CKPT_FORMAT, "ctx": learner.ctx.detach().cpu(), "class_names": learner.class_names,
                "n_ctx": learner.n_ctx, "csc": learner.csc, "ctx_init": learner.ctx_init, **meta}, path)


def read_checkpoint(path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") == CKPT_FORMAT:
        return ck
    if "prompts" in ck:  # the original learned_prompts.pth (2025-09)
        return {"format": "legacy", "prompts": ck["prompts"], "class_names": ck.get("class_names", ["fire", "non fire"]),
                "arch": "ViT-B-32", "pretrained": "openai", "resize_mode": "shortest", "quick_gelu": False,
                "threshold": None}
    raise SystemExit(f"unknown checkpoint format: {path}")


@torch.no_grad()
def classifier_weights(model, tokenizer, ck: dict, device) -> torch.Tensor:
    """(n_classes, d) normalised class weights for a checkpoint read with read_checkpoint."""
    if ck["format"] == "legacy":
        # Same formula the legacy model was TRAINED with: mean over [10 learned vectors, text embedding].
        # (The old predict script used a different formula, which is the train/predict mismatch it had.)
        e = model.encode_text(tokenizer(ck["class_names"]).to(device)).float()
        p = ck["prompts"].to(device).float()
        return F.normalize(torch.cat([p, e.unsqueeze(1)], dim=1).mean(1), dim=-1)
    learner = PromptLearner(model, tokenizer, ck["class_names"], ck["n_ctx"], ck["ctx_init"], ck["csc"])
    learner.ctx.data.copy_(ck["ctx"].to(device))
    return learner.class_weights(model)


def fire_prob(model, feats: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """P(fire) for normalised image features."""
    logits = model.logit_scale.exp().float() * feats.to(weights.device) @ weights.T
    return logits.softmax(-1)[:, FIRE].cpu()


# ---------------------------------------------------------------- metrics
def threshold_at_recall(p_fire: np.ndarray, is_fire: np.ndarray, target: float) -> float:
    """Highest threshold t such that at least `target` of the fire images have P(fire) >= t."""
    s = np.sort(p_fire[is_fire])[::-1]
    if len(s) == 0:
        return 0.5
    return float(s[max(0, math.ceil(target * len(s)) - 1)])


def operating_point(p_fire: np.ndarray, is_fire: np.ndarray, t: float) -> dict:
    pos, neg = p_fire[is_fire], p_fire[~is_fire]
    return {"threshold": t,
            "fire_recall": float((pos >= t).mean()) if len(pos) else float("nan"),
            "false_alarms_rejected": float((neg < t).mean()) if len(neg) else float("nan"),
            "n_fire": int(len(pos)), "n_not_fire": int(len(neg))}
