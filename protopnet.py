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

# ==================== Setup ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

wandb.login(key="")
wandb.init(project="siglip-protopnet", name="stage2a-protopnet-multilabel")

# ==================== Load Pretrained SigLIP2 ====================
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "ViT-SO400M-16-SigLIP2-512", pretrained="webli"
)

siglip_ckpt = torch.load("weights_finetune/finetune_epoch_8.pth")
model.load_state_dict(siglip_ckpt, strict=False)
print("✅ Loaded fine-tuned weights from Stage 1 (Epoch 8)")

# Freeze SigLIP2
for param in model.visual.parameters():
    param.requires_grad = False
print("✅ SigLIP2 backbone is frozen")

# ==================== ProtoPNet ====================
class ProtoPNet(nn.Module):
    def __init__(self, backbone, num_prototypes_per_class=6, num_classes=10):
        super().__init__()
        self.backbone = backbone.visual
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.prototype_vectors = nn.Parameter(
            torch.rand(num_classes * num_prototypes_per_class, 1152)
        )
        self.last_layer = nn.Linear(num_classes * num_prototypes_per_class, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        distances = torch.cdist(features.unsqueeze(1), self.prototype_vectors.unsqueeze(0))
        similarities = -distances.squeeze(1)
        logits = self.last_layer(similarities)
        return logits, similarities

proto_model = ProtoPNet(model).to(device)

# ==================== Dataset ====================
class CholecT45Dataset(Dataset):
    def __init__(self, json_file, preprocess, augment=False):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.preprocess = preprocess
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(BytesIO(base64.b64decode(item['image']))).convert("RGB")
        image_np = np.array(image)
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

train_dataset = CholecT45Dataset("ivt_train.json", preprocess_train, augment=True)
val_dataset = CholecT45Dataset("ivt_val.json", preprocess_val)
test_dataset = CholecT45Dataset("ivt_test.json", preprocess_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

# ==================== Loss and Optimizer ====================
criterion = nn.BCEWithLogitsLoss()

def prototype_loss(similarities, labels, num_classes=10, num_prototypes_per_class=6):
    batch_size = similarities.size(0)
    clustering_losses, separation_losses = [], []

    for i in range(batch_size):
        sample_labels = labels[i]
        sample_sims = similarities[i]
        pos_proto_indices, neg_proto_indices = [], []

        for class_idx in range(num_classes):
            proto_indices = list(range(class_idx * num_prototypes_per_class, (class_idx + 1) * num_prototypes_per_class))
            if sample_labels[class_idx] == 1:
                pos_proto_indices.extend(proto_indices)
            else:
                neg_proto_indices.extend(proto_indices)

        if pos_proto_indices:
            clustering_losses.append(-sample_sims[pos_proto_indices].mean())
        if neg_proto_indices:
            separation_losses.append(sample_sims[neg_proto_indices].mean())

    return torch.stack(clustering_losses).mean(), torch.stack(separation_losses).mean()

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, proto_model.parameters()), lr=1e-4, weight_decay=1e-4)
os.makedirs("weights_proto", exist_ok=True)

# ==================== Evaluation ====================
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

    return running_loss / len(loader), average_precision_score(np.concatenate(all_targets), np.concatenate(all_preds), average="macro")

# ==================== Training ====================
def train_proto_model(model, train_loader, val_loader, optimizer, num_epochs=30):
    best_val_mAP = 0.0
    for epoch in range(num_epochs):
        if epoch == 17:
            print("🔓 Unfreezing last 5 Transformer blocks of SigLIP2...")
            for block in model.backbone.transformer.resblocks[-5:]:
                for param in block.parameters():
                    param.requires_grad = True
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

        val_loss, val_mAP = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss={running_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}, Val mAP={val_mAP:.4f}")

        if val_mAP > best_val_mAP:
            best_val_mAP = val_mAP
            torch.save(model.state_dict(), "weights_proto/protopnet_stage2a.pth")
            torch.save(model.prototype_vectors.detach().cpu(), f"weights_proto/prototypes_epoch_{epoch+1}.pt")
            print("✅ Saved best ProtoPNet model and prototypes!")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss,
            "val_mAP": val_mAP
        })

# ==================== Run Training and Test ====================
train_proto_model(proto_model, train_loader, val_loader, optimizer, num_epochs=30)

print("📥 Loading best ProtoPNet model for final testing...")
proto_model.load_state_dict(torch.load("weights_proto/protopnet_stage2a.pth"))
test_loss, test_mAP = evaluate_model(proto_model, test_loader)
print(f"🧪 Final Test Loss: {test_loss:.4f}, Test mAP: {test_mAP:.4f}")

wandb.log({
    "final_test_loss": test_loss,
    "final_test_mAP": test_mAP
})
