# predict_clip_fire_improved.py
import os
import torch
import pandas as pd
from PIL import Image
import open_clip
from sklearn.metrics import classification_report, accuracy_score

# ------------------- SETTINGS -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
test_fire_path = r"C:\Users\K1000\Desktop\Dataset\test\fire"
test_nonfire_path = r"C:\Users\K1000\Desktop\Dataset\test\non_fire"
prompt_path = "learned_prompts.pth"
output_csv = "fire_predictions.csv"

# ------------------- LOAD MODEL -------------------
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.to(device)
model.eval()

# ------------------- LOAD LEARNED PROMPTS -------------------
learned_prompts = torch.load(prompt_path, map_location=device)
if isinstance(learned_prompts, dict):  # in case it was saved as dict
    learned_prompts = learned_prompts.get("prompts", None)
if learned_prompts is None:
    raise ValueError("❌ Could not load learned prompts correctly.")
learned_prompts = learned_prompts.to(device)

class_names = ["fire", "non fire"]

# Encode original text embeddings
text_tokens = tokenizer(class_names).to(device)
with torch.no_grad():
    class_embeddings = model.encode_text(text_tokens)

# Apply prompt tuning
def apply_prompt_tuning(class_embeddings, learned_prompts):
    tuned_embeddings = []
    for i, emb in enumerate(class_embeddings):
        # Ensure learned prompt is 2D [num_prompts, embed_dim]
        prompt = learned_prompts[i]
        if prompt.dim() == 3:  # sometimes saved with an extra batch dim
            prompt = prompt.squeeze(0)

        # Average prompt tokens -> single vector
        prompt_mean = prompt.mean(0, keepdim=True)  # [1, embed_dim]

        # Combine with original embedding
        tuned = torch.cat([prompt_mean, emb.unsqueeze(0)], dim=0).mean(0)

        tuned_embeddings.append(tuned)

    tuned_embeddings = torch.stack(tuned_embeddings, dim=0)
    tuned_embeddings = tuned_embeddings / tuned_embeddings.norm(dim=-1, keepdim=True)
    return tuned_embeddings


tuned_embeddings = apply_prompt_tuning(class_embeddings, learned_prompts)

# ------------------- PREDICTION -------------------
def predict_folder(folder, true_label, tuned_embeddings):
    results = []
    for img_name in os.listdir(folder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".tif")):
            continue
        img_path = os.path.join(folder, img_name)
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ tuned_embeddings.T
                probs = torch.softmax(logits, dim=-1)
                pred_idx = probs.argmax(dim=-1).item()
                confidence = probs[0, pred_idx].item()
                predicted_label = class_names[pred_idx]

            results.append({
                "image_name": img_name,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": round(confidence, 4)
            })
            print(f"✅ {img_name} | True: {true_label} | Pred: {predicted_label} ({confidence:.2f})")
        except Exception as e:
            print(f"❌ Error with {img_name}: {e}")
    return results

all_results = []
all_results += predict_folder(test_fire_path, "fire", tuned_embeddings)
all_results += predict_folder(test_nonfire_path, "non fire", tuned_embeddings)

# ------------------- SAVE CSV -------------------
df = pd.DataFrame(all_results)
df.to_csv(output_csv, index=False)
print(f"\n📁 Predictions saved to {output_csv}")

# ------------------- STATISTICS -------------------
y_true = df["true_label"]
y_pred = df["predicted_label"]
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
print(f"✅ Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
