# train_and_predict_pytorch.py
# Usage:
#   edit DATA_ROOT to point to the C-NMC_Leukemia folder
#   pip install torch torchvision scikit-learn pillow  (if not already installed)
#   python train_and_predict_pytorch.py

import os
import csv
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

from sklearn.metrics import roc_auc_score, accuracy_score

# -------------------------
# Config - update paths if needed
# -------------------------
DATA_ROOT = "./leukemia-classification/C-NMC_Leukemia"
TRAIN_ROOT = os.path.join(DATA_ROOT, "training_data")
VAL_ROOT = os.path.join(DATA_ROOT, "validation_data")   # unlabeled
TEST_ROOT = os.path.join(DATA_ROOT, "testing_data")     # unlabeled

OUTPUT_DIR = "./output_pytorch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 123
EPOCHS_HEAD = 8
EPOCHS_FINETUNE = 10
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
VAL_SPLIT = 0.10   # fraction of training used as labeled validation for monitoring

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Helpers to collect files
# -------------------------
def collect_labeled_images(root):
    paths = []
    labels = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(IMG_EXTS):
                continue
            full = os.path.join(dirpath, fname)
            parts = [p.lower() for p in full.split(os.sep)]
            if "all" in parts:
                label = 1
            elif "hem" in parts:
                label = 0
            else:
                continue
            paths.append(full)
            labels.append(label)
    # sort for determinism
    pairs = sorted(zip(paths, labels))
    if not pairs:
        return [], []
    paths, labels = zip(*pairs)
    return list(paths), list(labels)

def collect_unlabeled_images(root):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(IMG_EXTS):
                continue
            paths.append(os.path.join(dirpath, fname))
    paths = sorted(paths)
    return paths

train_paths, train_labels = collect_labeled_images(TRAIN_ROOT)
val_unlabeled_paths = collect_unlabeled_images(VAL_ROOT)
test_unlabeled_paths = collect_unlabeled_images(TEST_ROOT)

print(f"Found {len(train_paths)} labeled train images, {len(val_unlabeled_paths)} unlabeled val images, {len(test_unlabeled_paths)} unlabeled test images")
if len(train_paths) == 0:
    raise SystemExit("No training images found. Check TRAIN_ROOT path and folder names.")

# -------------------------
# PyTorch Dataset classes
# -------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

class LabeledImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

class UnlabeledImageDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, p

# -------------------------
# Make datasets and dataloaders (with small labeled val split)
# -------------------------
n_total = len(train_paths)
idxs = list(range(n_total))
random.shuffle(idxs)
n_val = max(1, int(n_total * VAL_SPLIT))
val_idxs = idxs[:n_val]
train_idxs = idxs[n_val:]

train_ds = LabeledImageDataset([train_paths[i] for i in train_idxs],
                               [train_labels[i] for i in train_idxs],
                               transform=train_transform)
val_ds_labeled = LabeledImageDataset([train_paths[i] for i in val_idxs],
                                     [train_labels[i] for i in val_idxs],
                                     transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds_labeled, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
val_unlabeled_loader = DataLoader(UnlabeledImageDataset(val_unlabeled_paths, transform=val_transform),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_unlabeled_loader = DataLoader(UnlabeledImageDataset(test_unlabeled_paths, transform=val_transform),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# compute counts for pos_weight
pos_count = sum(train_labels[i] for i in train_idxs)
neg_count = len(train_idxs) - pos_count
print("Train splits -> train:", len(train_idxs), "val(labelled):", len(val_idxs))
print("pos_count (train):", pos_count, "neg_count:", neg_count)

# -------------------------
# Build model (ResNet50), replace final fc with single logit output
# -------------------------
model = models.resnet50(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)   # single logit for binary classification
model = model.to(device)

# Loss: BCEWithLogitsLoss with pos_weight to balance classes (pos_weight = neg/pos)
pos_weight = torch.tensor([ (neg_count / max(1, pos_count)) ], dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_count>0 else nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR_HEAD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# -------------------------
# Training helpers
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    print("Training epoch")
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)  # shape (B,1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        all_preds.extend(probs.tolist())
        all_targets.extend(labels.detach().cpu().numpy().ravel().tolist())
    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.0
    acc = accuracy_score([int(x>=0.5) for x in all_targets], [int(x>=0.5) for x in all_preds])
    return avg_loss, auc, acc

def evaluate(model, loader, criterion, device):
    print("Evaluating Model")
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_preds.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().ravel().tolist())
    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.0
    acc = accuracy_score([int(x>=0.5) for x in all_targets], [int(x>=0.5) for x in all_preds])
    return avg_loss, auc, acc

# -------------------------
# Training: head then fine-tune
# -------------------------
best_auc = -1.0
best_path = os.path.join(OUTPUT_DIR, "best_resnet50.pth")

# Train head first (all layers trainable here for simplicity; you can freeze base layers if desired)
print("Starting head training...")
for epoch in range(EPOCHS_HEAD):
    train_loss, train_auc, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_auc)
    print(f"[Head] Epoch {epoch+1}/{EPOCHS_HEAD}  train_loss={train_loss:.4f} train_auc={train_auc:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}")
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), best_path)
        print("Saved best model (head) with val_auc:", best_auc)

# Fine-tune: optionally unfreeze entire model and lower lr
print("Fine-tuning full model...")
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

for epoch in range(EPOCHS_FINETUNE):
    train_loss, train_auc, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_auc)
    print(f"[Fine] Epoch {epoch+1}/{EPOCHS_FINETUNE}  train_loss={train_loss:.4f} train_auc={train_auc:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}")
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), best_path)
        print("Saved best model (fine) with val_auc:", best_auc)

print("Training finished. Best val_auc:", best_auc)

# -------------------------
# Load best model and run inference on unlabeled sets
# -------------------------
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

def predict_and_save(loader, out_csv):
    rows = []
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc="Predicting"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            for p, prob in zip(paths, probs):
                # write relative path from DATA_ROOT if desired:
                rel = os.path.relpath(p, DATA_ROOT)
                label = "all" if prob >= 0.5 else "hem"
                rows.append((rel, label, float(prob)))
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_label", "probability_all"])
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} predictions to {out_csv}")

predict_and_save(val_unlabeled_loader, os.path.join(OUTPUT_DIR, "predictions_validation.csv"))
predict_and_save(test_unlabeled_loader, os.path.join(OUTPUT_DIR, "predictions_testing.csv"))

# Save final model (scripted)
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "resnet50_final.pth"))
print("Saved final model to", os.path.join(OUTPUT_DIR, "resnet50_final.pth"))