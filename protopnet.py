# === Stage 2A: ProtoPNet with Frozen SigLIP2 - Multilabel Classification ===

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import open_clip
import os
from tqdm import tqdm
import numpy as np
import json
from io import BytesIO
from PIL import Image
import base64
import cv2
from sklearn.metrics import average_precision_score
import torch.backends.cudnn as cudnn
import wandb

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# ✅ Weights & Biases setup
wandb.login(key="a8832575b3abf5340a76141f84f38cb6c1c19247")
wandb.init(project="siglip-protopnet", name="stage2a-protopnet-multilabel")

# ✅ Load pretrained SigLIP2 model
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "ViT-SO400M-16-SigLIP2-512", pretrained="webli"
)

# ✅ Load Stage 1 fine-tuned weights
siglip_ckpt = torch.load("weights_finetune/finetune_epoch_8.pth")
model.load_state_dict(siglip_ckpt, strict=False)
print("✅ Loaded fine-tuned weights from Stage 1 (Epoch 8)")

# ✅ Freeze entire SigLIP2 visual backbone initially
for param in model.visual.parameters():
    param.requires_grad = False
print("✅ SigLIP2 backbone is frozen")

# ✅ Define ProtoPNet architecture
class ProtoPNet(nn.Module):
    def __init__(self, backbone, num_prototypes_per_class=6, num_classes=10):
        super().__init__()
        self.backbone = backbone.visual  # Frozen SigLIP2 visual encoder
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class

        # Learnable prototype vectors (one per prototype)
        self.prototype_vectors = nn.Parameter(
            torch.rand(num_classes * num_prototypes_per_class, 1152)  # Match SigLIP2 feature size
        )

        # Final classification layer (linear over similarities)
        self.last_layer = nn.Linear(num_classes * num_prototypes_per_class, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # (B, 1152)
        distances = torch.cdist(features.unsqueeze(1), self.prototype_vectors.unsqueeze(0))  # (B, 1, 60)
        similarities = -distances.squeeze(1)  # (B, 60), negative distance = similarity
        logits = self.last_layer(similarities)  # (B, 10)
        return logits, similarities

# ✅ Initialize ProtoPNet with frozen SigLIP2 backbone
proto_model = ProtoPNet(model).to(device)

# ✅ Define CholecT45 Dataset with CLAHE preprocessing
class CholecT45Dataset(Dataset):
    def __init__(self, json_file, preprocess, augment=False):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.data = [item for item in data if 'image' in item and 'verb_labels' in item]
        self.preprocess = preprocess
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(BytesIO(base64.b64decode(item['image']))).convert("RGB")
        image_np = np.array(image)

        # ✅ Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        image_pil = Image.fromarray(image_np)
        image_final = self.preprocess(image_pil)

        labels = torch.tensor(item['verb_labels'], dtype=torch.float32)
        return image_final, labels

# ✅ Load datasets and dataloaders
train_dataset = CholecT45Dataset("../../../../instrument_verb_train.json", preprocess_train, augment=True)
val_dataset = CholecT45Dataset("../../../../instrument_verb_val.json", preprocess_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

# ✅ Define binary cross-entropy loss
criterion = nn.BCEWithLogitsLoss()

# ✅ Prototype clustering + separation loss
def prototype_loss(similarities, labels, num_classes=10, num_prototypes_per_class=6):
    batch_size = similarities.size(0)
    clustering_losses, separation_losses = [], []

    for i in range(batch_size):
        sample_labels = labels[i]  # (10,)
        sample_sims = similarities[i]  # (60,)
        pos_proto_indices, neg_proto_indices = [], []

        for class_idx in range(num_classes):
            proto_indices = list(range(class_idx * num_prototypes_per_class,
                                       (class_idx + 1) * num_prototypes_per_class))
            if sample_labels[class_idx] == 1:
                pos_proto_indices.extend(proto_indices)  # Pull prototypes closer
            else:
                neg_proto_indices.extend(proto_indices)  # Push prototypes away

        if pos_proto_indices:
            clustering_losses.append(-sample_sims[pos_proto_indices].mean())
        if neg_proto_indices:
            separation_losses.append(sample_sims[neg_proto_indices].mean())

    clustering_loss = torch.stack(clustering_losses).mean()
    separation_loss = torch.stack(separation_losses).mean()
    return clustering_loss, separation_loss

# ✅ Optimizer for only currently trainable parameters
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, proto_model.parameters()), lr=1e-4, weight_decay=1e-4)

# ✅ Make folder for saving weights
os.makedirs("weights_proto", exist_ok=True)

# ✅ Evaluation function
def evaluate_model(model, loader):
    model.eval()
    all_targets, all_preds = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            all_targets.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(logits).cpu().numpy())

    avg_loss = running_loss / len(loader)
    mAP = average_precision_score(np.concatenate(all_targets), np.concatenate(all_preds), average="macro")
    return avg_loss, mAP

# ✅ Main training loop
def train_proto_model(model, train_loader, val_loader, optimizer, num_epochs=30):
    best_val_mAP = 0.0

    for epoch in range(num_epochs):
        # 🔓 Unfreeze last 5 transformer blocks after 17 epochs
        if epoch == 17:
            print("🔓 Unfreezing last 5 transformer blocks of SigLIP2...")
            for block in model.backbone.transformer.resblocks[-5:]:
                for param in block.parameters():
                    param.requires_grad = True
            print("🔄 Reinitializing optimizer to include unfrozen layers...")
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)

        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, similarities = model(images)

            bce_loss = criterion(logits, labels)
            cluster_loss, sep_loss = prototype_loss(similarities, labels)

            total_loss = bce_loss + 0.8 * cluster_loss + 0.1 * sep_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        # ✅ Evaluate on validation set
        val_loss, val_mAP = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss={running_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}, Val mAP={val_mAP:.4f}")

        # ✅ Save best model and prototypes
        if val_mAP > best_val_mAP:
            best_val_mAP = val_mAP
            torch.save(model.state_dict(), "weights_proto/protopnet_stage2a.pth")
            torch.save(model.prototype_vectors.detach().cpu(), f"weights_proto/prototypes_epoch_{epoch+1}.pt")
            print("✅ Saved best ProtoPNet model and prototypes!")

        # ✅ Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss,
            "val_mAP": val_mAP
        })

# ✅ Start training
train_proto_model(proto_model, train_loader, val_loader, optimizer, num_epochs=30)
