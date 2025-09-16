import os, glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import open_clip
from tqdm import tqdm

# ------------------- CONFIG -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
num_epochs = 20
lr = 1e-3
prompt_length = 10
val_split = 0.2       # use 20% for validation
fire_dir = os.getenv("TRAIN_FIRE_PATH")
non_fire_dir = os.getenv("TRAIN_NONFIRE_PATH")

# ------------------- LOAD MODEL -------------------
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)

# ------------------- DATASET -------------------
valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.gif", "*.tif")

def get_image_paths(folder):
    files = []
    for ext in valid_exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return files

fire_images = get_image_paths(fire_dir)
non_fire_images = get_image_paths(non_fire_dir)

# --- Sanity check
if len(fire_images) == 0 or len(non_fire_images) == 0:
    raise ValueError(f"🔥 Dataset issue: fire={len(fire_images)} | non-fire={len(non_fire_images)}. "
                     f"Make sure both folders contain images.")

all_images = fire_images + non_fire_images
all_labels = [0]*len(fire_images) + [1]*len(non_fire_images)  # 0=fire,1=non-fire

class FireSmokeDataset(Dataset):
    def __init__(self, image_paths, labels, preprocess):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.preprocess(img), torch.tensor(self.labels[idx])

# Create dataset
dataset = FireSmokeDataset(all_images, all_labels, preprocess)

# Stratified split for train/val
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------- PROMPT TUNING -------------------
embedding_dim = model.text_projection.shape[1]
class_names = ["fire", "non fire"]

# One soft prompt per class
learned_prompts = nn.Parameter(
    torch.randn(len(class_names), prompt_length, embedding_dim, device=device)
)

text_tokens = tokenizer(class_names).to(device)
with torch.no_grad():
    class_embeddings = model.encode_text(text_tokens)

def apply_prompt_tuning(class_embeddings, learned_prompts):
    tuned_embeddings = []
    for i, emb in enumerate(class_embeddings):
        tuned = torch.cat([learned_prompts[i], emb.unsqueeze(0)], dim=0)
        tuned = tuned.mean(0)
        tuned_embeddings.append(tuned)
    tuned_embeddings = torch.stack(tuned_embeddings, dim=0)
    tuned_embeddings = tuned_embeddings / tuned_embeddings.norm(dim=-1, keepdim=True)
    return tuned_embeddings

# ------------------- TRAINING SETUP -------------------
class_counts = [len(fire_images), len(non_fire_images)]
total_count = sum(class_counts)
weights = [total_count/c if c > 0 else 0.0 for c in class_counts]
weights = torch.tensor(weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam([learned_prompts], lr=lr) # new_prompt = old_prompt − lr × gradient

# ------------------- TRAINING LOOP -------------------
for epoch in range(num_epochs):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        tuned_embeddings = apply_prompt_tuning(class_embeddings, learned_prompts)
        logits = 100.0 * image_features @ tuned_embeddings.T

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = logits.max(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct/total)

    # Validation
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            tuned_embeddings = apply_prompt_tuning(class_embeddings, learned_prompts)
            logits = 100.0 * image_features @ tuned_embeddings.T
            _, preds = logits.max(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total if val_total > 0 else 0
    print(f"📊 Epoch {epoch+1}/{num_epochs} | Train Acc: {correct/total:.4f} | Val Acc: {val_acc:.4f}")

# ------------------- SAVE PROMPTS -------------------
torch.save({
    "prompts": learned_prompts.detach().cpu(),
    "class_names": class_names,
    "config": {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "prompt_length": prompt_length
    }
}, "learned_prompts.pth")

print("✅ Saved learned prompts and config to learned_prompts.pth")
