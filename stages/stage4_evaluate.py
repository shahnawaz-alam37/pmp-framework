"""
Stage 4: Evaluation
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
        predictions = torch.argmin(distances, dim=1)
        return {'predictions': predictions}


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


def evaluate_shot(model, dataset, device, k_shot, n_way=5, episodes=150):
    loader = FewShotDataLoader(dataset, n_way=n_way, k_shot=k_shot, 
                           query_samples=5, episodes_per_epoch=episodes, num_workers=2, batch_size=1)
    
    protonet = ProtoNetFewShot(model, n_way=n_way).to(device)
    protonet.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for episode in loader:
            data = load_episode(episode, loader.dataset, device)
            if data[0] is None:
                continue
            
            support_images, support_labels, query_images, query_labels = data
            outputs = protonet(support_images, support_labels, query_images)
            predictions = outputs['predictions']
            correct += (predictions == query_labels).sum().item()
            total += query_labels.size(0)
    
    return correct / total if total > 0 else 0


def main():
    print("=" * 60)
    print("Stage 4: Evaluation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = cfg.get_config()
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / config['dataset']['data_dir']
    
    checkpoint_path = Path('checkpoints/stage3_model.pth')
    if not checkpoint_path.exists():
        checkpoint_path = PROJECT_ROOT / 'checkpoints/stage3_model.pth'
    
    if not checkpoint_path.exists():
        print("ERROR: Model not found!")
        return
    
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    num_classes = checkpoint.get('num_classes', config['dataset']['num_classes'])
    model = PrunedResNet18(num_classes=num_classes, pretrained=False)

    state_dict = (
        checkpoint.get('compressed_backbone_state_dict')
        or checkpoint.get('backbone_state_dict')
        or checkpoint.get('model_state_dict')
        or {}
    )

    if not state_dict:
        print("ERROR: No valid state_dict found in checkpoint")
        return

    model_state = model.backbone.state_dict()
    compatible = {
        k: v for k, v in state_dict.items()
        if k in model_state and torch.is_tensor(v) and model_state[k].shape == v.shape
    }

    model_state.update(compatible)
    model.backbone.load_state_dict(model_state)

    print(f"Loaded compatible tensors: {len(compatible)}/{len(state_dict)}")
    if len(compatible) == 0:
        print("ERROR: No compatible tensors could be loaded from checkpoint")
        return

    model = model.to(device)
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = PlantDiseaseDataset(DATA_DIR, split='val', transform=val_transform, image_size=224)
    if len(val_dataset) == 0:
        print(f"ERROR: Validation dataset is empty at {DATA_DIR / 'val'}")
        return
    print(f"Dataset: {len(val_dataset)} samples, {len(val_dataset.classes)} classes")

    print(f"\nModel size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
    
    print(f"\n{'='*60}")
    print("Evaluating Few-Shot Performance")
    print(f"{'='*60}")
    
    results = {}
    for shot_name, k_shot in [('1_shot', 1), ('5_shot', 5), ('10_shot', 10)]:
        print(f"\n--- {shot_name} ---")
        acc = evaluate_shot(model, val_dataset, device, k_shot)
        print(f"Accuracy: {acc:.2%}")
        results[shot_name] = acc
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"1-shot:  {results.get('1_shot', 0):.2%}")
    print(f"5-shot:  {results.get('5_shot', 0):.2%}")
    print(f"10-shot: {results.get('10_shot', 0):.2%}")
    
    os.makedirs(PROJECT_ROOT / 'results', exist_ok=True)
    with open(PROJECT_ROOT / 'results/final_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/final_evaluation.json")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()