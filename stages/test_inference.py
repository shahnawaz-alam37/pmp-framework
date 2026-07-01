"""
Explain: Few-shot vs Full Classification
========================================

This model is trained for FEW-SHOT LEARNING.
That means:
- You show it 1-5 examples of EACH disease
- Then it classifies new images

It does NOT work like regular classifiers where you just show 1 image.

HOW TO USE THIS MODEL:
====================

Option 1: ProtoNet (for new diseases you don't have)
1. Collect 5 photos of the NEW disease
2. Use those as "support set" 
3. Show new photo → it classifies based on similarity

Option 2: Full Classifier (if you need simple one-image)
Use a standard classifier like this quick tool.
"""

print(__doc__)


def create_and_train_classifier():
    """Quick full classifier for single-image inference"""
    
    print("Creating fresh classifier (standard ResNet-18)...")
    
    import torch
    import torch.nn as nn
    from torchvision import models
    from torch.utils.data import DataLoader
    from pathlib import Path
    import random
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data_filtered'
    
    # Model
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, 41)
    
    # Use existing full model if available
    full_model_path = PROJECT_ROOT / 'checkpoints/classifier_full.pth'
    if full_model_path.exists():
        print("Loading existing full classifier...")
        model.load_state_dict(torch.load(full_model_path))
        return model
    
    # Transform
    from torchvision import transforms
    from dataset import PlantDiseaseDataset
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = PlantDiseaseDataset(DATA_DIR, split='train', transform=train_transform)
    
    # Quick subset
    n = 3000
    indices = random.sample(range(len(train_dataset)), n)
    
    class Subset:
        def __init__(self, ds, idx): self.ds, self.idx = ds, idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i): return self.ds[self.idx[i]]
    
    loader = DataLoader(Subset(train_dataset, indices), batch_size=32, shuffle=True)
    
    print(f"Quick training on {n} images...")
    model.fc = nn.Linear(512, 41)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(3):
        model.train()
        for imgs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
    
    # Save
    torch.save(model.state_dict(), full_model_path)
    print(f"Saved to {full_model_path}")
    
    return model


if __name__ == '__main__':
    print("=" * 60)
    print("ANSWER TO YOUR QUESTION:")
    print("=" * 60)
    print("""
This model is for FEW-SHOT learning. 
To classify one image:

FOR NEW DISEASES (few-shot style):
- Take 5 photos of each disease  
- Use as support set
- Query new images

FOR SIMPLE INFERENCE:
- Need full classifier (not yet trained)

Would you like me to train a proper full classifier?
(Will take ~5-10 minutes)
""")
    
    answer = input("Train full classifier now? (y/n): ").strip().lower()
    if answer == 'y':
        model = create_and_train_classifier()