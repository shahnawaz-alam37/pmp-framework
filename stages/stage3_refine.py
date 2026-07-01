"""
Stage 3: Refinement Pruning + Final Fine-tune
"""

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from dataset import PlantDiseaseDataset, FewShotDataLoader
from stages.stage1_prune import PrunedResNet18


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / cfg.DATASET['data_dir']


class ProtoNetFewShot(nn.Module):
    def __init__(self, backbone, n_way=5):
        super().__init__()
        self.backbone = backbone
        self.n_way = n_way
        self.feature_extractor = nn.Sequential(*list(backbone.backbone.children())[:-1])
    
    def compute_prototypes(self, support_images, support_labels):
        features = self.feature_extractor(support_images)
        features = features.view(features.size(0), -1)
        prototypes = torch.zeros(self.n_way, features.size(1), device=features.device)
        for c in range(self.n_way):
            mask = (support_labels == c)
            if mask.sum() > 0:
                prototypes[c] = features[mask].mean(dim=0)
        return prototypes
    
    def forward(self, support_images, support_labels, query_images):
        prototypes = self.compute_prototypes(support_images, support_labels)
        query_features = self.feature_extractor(query_images)
        query_features = query_features.view(query_features.size(0), -1)
        distances = (query_features.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=2)
        logits = -distances
        predictions = torch.argmin(distances, dim=1)
        return {'logits': logits, 'predictions': predictions}


def build_compressed_backbone_state(checkpoint_state: dict, runtime_state: dict) -> dict:
    """Keep compressed tensors from previous stage and refresh compatible classifier tensors."""
    save_state = {}

    for k, v in checkpoint_state.items():
        if torch.is_tensor(v):
            save_state[k] = v.detach().cpu().clone()
        else:
            save_state[k] = v

    for head_key in ('fc.weight', 'fc.bias'):
        if head_key in runtime_state:
            head_tensor = runtime_state[head_key].detach().cpu().clone()
            if head_key in save_state:
                if torch.is_tensor(save_state[head_key]) and save_state[head_key].shape == head_tensor.shape:
                    save_state[head_key] = head_tensor
            else:
                save_state[head_key] = head_tensor

    return save_state


def load_episode(episode, dataset, device):
    items = episode.support_data
    if len(items) < 5:
        return None, None, None, None
    
    imgs, lbls = [], []
    for i in range(len(items)):
        idx, lbl = items[i]
        imgs.append(dataset[idx][0])
        lbls.append(lbl)
    support_images = torch.stack(imgs).to(device)
    support_labels = torch.tensor(lbls, dtype=torch.long).to(device)
    
    items = episode.query_data
    if len(items) < 5:
        return None, None, None, None
    
    imgs, lbls = [], []
    for i in range(len(items)):
        idx, lbl = items[i]
        imgs.append(dataset[idx][0])
        lbls.append(lbl)
    query_images = torch.stack(imgs).to(device)
    query_labels = torch.tensor(lbls, dtype=torch.long).to(device)
    
    return support_images, support_labels, query_images, query_labels


def validate(model, loader, device, n_way):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for episode in loader:
            data = load_episode(episode, loader.dataset, device)
            if data[0] is None:
                continue
            
            support_images, support_labels, query_images, query_labels = data
            outputs = model(support_images, support_labels, query_images)
            predictions = outputs['predictions']
            correct += (predictions == query_labels).sum().item()
            total += query_labels.size(0)
    
    return correct / total if total > 0 else 0


def main():
    print("=" * 60)
    print("Stage 3: Refinement Pruning + Final Fine-tune")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = cfg.get_config()
    
    # Load from Stage 2 checkpoint
    prev_checkpoint = PROJECT_ROOT / 'checkpoints/stage2_model.pth'
    
    if not prev_checkpoint.exists():
        print("ERROR: Stage 2 checkpoint not found!")
        return
    
    print(f"\nLoading from Stage 2: {prev_checkpoint.name}")
    checkpoint = torch.load(prev_checkpoint, map_location=device)
    
    num_classes = checkpoint.get('num_classes', 41)
    compressed_sd = checkpoint.get('compressed_backbone_state_dict')
    if compressed_sd is None:
        compressed_sd = checkpoint.get('backbone_state_dict', {})
    if not compressed_sd:
        print("ERROR: Stage 2 checkpoint missing backbone_state_dict!")
        return

    model = PrunedResNet18(num_classes=num_classes, pretrained=False)

    runtime_sd = model.backbone.state_dict()
    loaded_layers = 0
    skipped_layers = 0
    for k, v in compressed_sd.items():
        if k in runtime_sd and runtime_sd[k].shape == v.shape:
            runtime_sd[k].copy_(v)
            loaded_layers += 1
        else:
            skipped_layers += 1

    model.backbone.load_state_dict(runtime_sd)
    model = model.to(device)
    
    print(f"Loaded pruned backbone from Stage 2")
    print(f"State tensors loaded: {loaded_layers}, skipped (shape mismatch): {skipped_layers}")
    
    # Calculate current compression
    total_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"Total parameters: {total_params:,}")
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = PlantDiseaseDataset(DATA_DIR, split='train', transform=train_transform, image_size=224)
    val_dataset = PlantDiseaseDataset(DATA_DIR, split='val', transform=val_transform, image_size=224)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f"ERROR: Dataset not loaded correctly from {DATA_DIR}")
        print("Expected folders: data_filtered/train and data_filtered/val")
        return
    
    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"Classes: {len(train_dataset.classes)} train, {len(val_dataset.classes)} val")
    
    train_loader = FewShotDataLoader(
        train_dataset, n_way=5, k_shot=5, query_samples=5,
        episodes_per_epoch=10, num_workers=2, batch_size=1
    )
    
    val_loader = FewShotDataLoader(
        val_dataset, n_way=5, k_shot=5, query_samples=5,
        episodes_per_epoch=10, num_workers=2, batch_size=1
    )
    
    # Fine-tune
    print(f"\n{'='*60}")
    print("Fine-tuning classifier")
    print(f"{'='*60}")
    
    protonet = ProtoNetFewShot(model, n_way=5).to(device)
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(protonet.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        protonet.train()
        running_loss = 0.0
        n_batches = 0

        for episode in train_loader:
            data = load_episode(episode, train_loader.dataset, device)
            if data[0] is None:
                continue
            
            support_images, support_labels, query_images, query_labels = data
            optimizer.zero_grad()
            outputs = protonet(support_images, support_labels, query_images)
            loss = criterion(outputs['logits'], query_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch+1}/5 - Loss: {running_loss / max(n_batches, 1):.4f}")
    
    val_accuracy = validate(protonet, val_loader, device, 5)
    print(f"Validation Accuracy: {val_accuracy:.2%}")
    
    print(f"\n{'='*60}")
    print("Saving checkpoint")
    print(f"{'='*60}")

    compressed_backbone_state = build_compressed_backbone_state(
        compressed_sd,
        model.backbone.state_dict(),
    )
    
    os.makedirs(PROJECT_ROOT / 'checkpoints', exist_ok=True)
    torch.save({
        'epoch': 0,
        'backbone_state_dict': compressed_backbone_state,
        'compressed_backbone_state_dict': compressed_backbone_state,
        'num_classes': num_classes,
        'val_accuracy': val_accuracy,
        'config': config,
        'dataset_info': {
            'data_dir': str(DATA_DIR),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'num_classes': len(train_dataset.classes),
        },
        'stage': 3
    }, PROJECT_ROOT / 'checkpoints/stage3_model.pth')
    
    size = (PROJECT_ROOT / 'checkpoints/stage3_model.pth').stat().st_size / (1024*1024)
    print(f"Saved to checkpoints/stage3_model.pth ({size:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("Stage 3 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()