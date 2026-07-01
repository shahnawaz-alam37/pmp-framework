"""
Stage 1: DACIS Pruning + Brief Fine-tuning
Initial channel pruning with DACIS scoring followed by brief fine-tuning
to recover accuracy.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from dataset import PlantDiseaseDataset, FewShotSampler, FewShotDataLoader

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / cfg.DATASET['data_dir']


class PrunedResNet18(nn.Module):
    """ResNet-18 with prunable channels."""
    
    def __init__(self, num_classes: int = 41, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        self.backbone.fc = nn.Linear(512, num_classes)
        self.num_classes = num_classes
        self.feature_dim = 512
        self.channel_masks = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_feature_extractor(self):
        return nn.Sequential(*list(self.backbone.children())[:-1])


def apply_pruning_hard(model: nn.Module, masks: dict, num_classes: int = 41) -> 'PrunedResNet18':
    """Hard pruning - rebuild model with only surviving channels."""
    print("Applying hard pruning - rebuilding model...")
    
    original = model.backbone
    
    in_masks = {}
    for name, mask in masks.items():
        in_masks[name] = mask.bool()
    
    total_original = 0
    total_pruned = 0
    
    for name, module in original.named_modules():
        if isinstance(module, nn.Conv2d):
            for mask_name, mask in in_masks.items():
                if mask_name in name and mask.numel() == module.weight.shape[1]:
                    keep_indices = torch.where(mask)[0]
                    orig_channels = module.weight.shape[1]
                    kept_channels = len(keep_indices)
                    
                    if kept_channels > 0 and kept_channels < orig_channels:
                        module.weight.data = module.weight.data[:, keep_indices, :, :].clone()
                        total_original += orig_channels
                        total_pruned += kept_channels
    
    compression = (total_original - total_pruned) / total_original if total_original > 0 else 0
    actual_removed = total_original - total_pruned
    print(f"Pruned: {total_original} -> {total_pruned} channels ({compression:.1%} removed)")
    
    model.backbone.fc = nn.Linear(512, num_classes)
    model.channel_masks = masks
    
    return model


def compute_dacis_scores(model: nn.Module, loader: FewShotDataLoader, device: torch.device) -> dict:
    """Compute DACIS importance scores."""
    print("Computing DACIS scores...")
    
    model.train()
    gradient_importance = {}
    
    n_collected = 0
    n_failed = 0
    
    for batch_idx, episode in enumerate(loader):
        if batch_idx >= 30:
            break
        
        try:
            items = episode.support_data
            if not items or len(items) < 5:
                n_failed += 1
                continue
            
            images = []
            labels = []
            for i in range(min(5, len(items))):
                idx, label = items[i]
                img, lbl = loader.dataset[idx]
                images.append(img)
                labels.append(lbl)
            
            support_images = torch.stack(images).to(device)
            support_labels = torch.tensor(labels, dtype=torch.long).to(device)
        except Exception as e:
            n_failed += 1
            continue
        
        model.zero_grad()
        
        try:
            output = model(support_images)
            support_labels_adjusted = support_labels % output.shape[1]
            loss = torch.nn.functional.cross_entropy(output, support_labels_adjusted)
            loss.backward()
            n_collected += 1
        except Exception as e:
            n_failed += 1
            continue
        
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                try:
                    grad = torch.abs(module.weight.grad.data)
                    weight = torch.abs(module.weight.data)
                    importance = (grad * weight).mean(dim=(1, 2, 3))
                    
                    if name not in gradient_importance:
                        gradient_importance[name] = []
                    gradient_importance[name].append(importance)
                except:
                    pass
    
    print(f"  Collected: {n_collected}, Failed: {n_failed}")
    
    dacis_scores = {}
    for name, grads in gradient_importance.items():
        if len(grads) > 0:
            stacked = torch.stack(grads).mean(dim=0)
            dacis_scores[name] = stacked
    
    print(f"Computed DACIS scores for {len(dacis_scores)} layers: {list(dacis_scores.keys())}")
    return dacis_scores
    gradient_importance[name].append(importance)
    
    dacis_scores = {}
    for name, grads in gradient_importance.items():
        stacked = torch.stack(grads).mean(dim=0)
        dacis_scores[name] = stacked
    
    print(f"Computed DACIS scores for {len(dacis_scores)} layers")
    return dacis_scores


def compute_pruning_masks(dacis_scores: dict, prune_ratio: float) -> dict:
    """Compute binary pruning masks based on DACIS scores."""
    masks = {}
    
    for name, scores in dacis_scores.items():
        if scores.numel() == 0:
            continue
        
        n_keep = int(scores.numel() * (1 - prune_ratio))
        if n_keep < 1:
            n_keep = 1
        
        _, indices = torch.topk(scores, n_keep)
        mask = torch.zeros(scores.numel())
        mask[indices] = 1.0
        masks[name] = mask
    
    return masks


def fine_tune_model(model: nn.Module, loader: FewShotDataLoader, device: torch.device,
                  epochs: int = 3) -> float:
    """Brief fine-tuning after pruning."""
    print(f"Fine-tuning for {epochs} epochs...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    n_batches = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_batches = 0
        
        for episode in loader:
            try:
                images = torch.stack([s[0] for s in episode.support_data]).to(device)
                labels = torch.tensor([s[1] for s in episode.support_data]).to(device)
            except:
                continue
            
            if images.size(0) < 5:
                continue
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        total_loss += avg_loss
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def validate_model(model: nn.Module, loader: FewShotDataLoader, device: torch.device) -> float:
    """Validate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for episode in loader:
            try:
                support_images = torch.stack([s[0] for s in episode.support_data]).to(device)
                support_labels = torch.tensor([s[1] for s in episode.support_data]).to(device)
                query_images = torch.stack([s[0] for s in episode.query_data]).to(device)
                query_labels = torch.tensor([s[1] for s in episode.query_data]).to(device)
            except:
                continue
            
            if support_images.size(0) < 5 or query_images.size(0) < 5:
                continue
            
            features = model.get_feature_extractor()(support_images)
            prototypes = []
            for class_idx in range(5):
                class_mask = (support_labels == class_idx)
                if class_mask.sum() > 0:
                    prototypes.append(features[class_mask].mean(dim=0))
            prototypes = torch.stack(prototypes)
            
            query_features = model.get_feature_extractor()(query_images)
            
            distances = (query_features.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=2)
            predictions = torch.argmin(distances, dim=1)
            
            correct += (predictions == query_labels).sum().item()
            total += query_labels.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    print("=" * 60)
    print("Stage 1: DACIS Pruning + Brief Fine-tuning")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint_path = Path('checkpoints/stage1_model.pth')
    config = cfg.get_config()
    
    if checkpoint_path.exists():
        print(f"\nLoading existing checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = PrunedResNet18(num_classes=checkpoint.get('num_classes', 41), pretrained=False)
        
        if 'backbone_state_dict' in checkpoint:
            model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from epoch {checkpoint.get('epoch', '?')}")
    else:
        print("\nCreating new model with ImageNet pretrained weights...")
        model = PrunedResNet18(num_classes=config['dataset']['num_classes'])
    
    model = model.to(device)
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = PlantDiseaseDataset(
        DATA_DIR,
        split='train',
        transform=train_transform,
        image_size=224
    )
    
    val_dataset = PlantDiseaseDataset(
        DATA_DIR,
        split='val',
        transform=val_transform,
        image_size=224
    )
    
    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"Classes: {len(train_dataset.classes)}")
    
    if len(train_dataset) == 0:
        print("ERROR: No training data found!")
        return
    
    train_loader = FewShotDataLoader(
        train_dataset,
        n_way=5,
        k_shot=5,
        query_samples=5,
        episodes_per_epoch=20,
        num_workers=2,
        batch_size=1
    )
    
    val_loader = FewShotDataLoader(
        val_dataset,
        n_way=5,
        k_shot=5,
        query_samples=5,
        episodes_per_epoch=20,
        num_workers=2,
        batch_size=1
    )
    
    prune_ratio = 0.35  # Hardcoded to match config
    
    print(f"\n{'='*60}")
    print(f"Computing DACIS scores for pruning ({prune_ratio:.0%} prune ratio)")
    print(f"{'='*60}")
    
    dacis_scores = compute_dacis_scores(model, train_loader, device)
    pruning_masks = compute_pruning_masks(dacis_scores, prune_ratio)
    
    print(f"\nPruning {len(pruning_masks)} layers...")
    for name, mask in pruning_masks.items():
        pruned = (mask == 0).sum().item()
        kept = (mask == 1).sum().item()
        print(f"  {name}: {pruned} pruned, {kept} kept ({pruned/(pruned+kept):.1%})")
    
    model = apply_pruning_hard(model, pruning_masks, num_classes=config['dataset']['num_classes'])
    model = model.to(device)
    
    print(f"\n{'='*60}")
    print("Fine-tuning after pruning")
    print(f"{'='*60}")
    
    fine_tune_model(model, train_loader, device, epochs=3)
    
    val_accuracy = validate_model(model, val_loader, device)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    
    print(f"\n{'='*60}")
    print("Saving checkpoint")
    print(f"{'='*60}")
    
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'epoch': 0,
        'backbone_state_dict': model.backbone.state_dict(),
        'num_classes': model.num_classes,
        'pruning_masks': pruning_masks,
        'val_accuracy': val_accuracy,
        'config': config,
        'stage': 1
    }, checkpoint_path)
    
    file_size = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"Saved to {checkpoint_path} ({file_size:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("Stage 1 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()