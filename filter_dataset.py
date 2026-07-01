"""
Filter PlantVillage dataset to selected plants only.
Plants: Tomato, Potato, Pepper, Strawberry, Corn, Apple, Cherry, Grape
"""

import shutil
import random
from pathlib import Path
import json


def filter_dataset():
    """Filter dataset to only include selected plants."""
    
    source_dir = Path("data")
    target_dir = Path("data_filtered")
    
    # Selected plant prefixes
    selected_plants = [
        "Tomato", "Potato", "Pepper", "Strawberry",
        "Corn", "Apple", "Cherry", "Grape"
    ]
    
    # Create target directories
    train_dir = target_dir / "train"
    val_dir = target_dir / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(42)
    
    total_train = 0
    total_val = 0
    classes = []
    
    # Check train directory
    source_train = source_dir / "train"
    if not source_train.exists():
        print("Source data not found. Run setup_dataset.py first.")
        return
    
    # Process each class
    for class_dir in sorted(source_train.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Check if class belongs to selected plants
        is_selected = any(class_name.startswith(plant) for plant in selected_plants)
        
        if not is_selected:
            print(f"Skipping: {class_name}")
            continue
        
        classes.append(class_name)
        print(f"Processing: {class_name}")
        
        # Get all images
        images = list(class_dir.glob("*.jpg"))
        if len(images) == 0:
            images = list(class_dir.glob("*.JPG"))
        
        if len(images) == 0:
            continue
        
        # Check val source
        source_val = source_dir / "val" / class_name
        val_images = list(source_val.glob("*.jpg")) if source_val.exists() else []
        
        # If val images don't exist, split from train
        if len(val_images) == 0:
            random.shuffle(images)
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
        else:
            train_images = images
            val_images = list(source_val.glob("*.png")) + list(source_val.glob("*.JPG")) + val_images
        
        # Copy train images
        train_class = train_dir / class_name
        train_class.mkdir(exist_ok=True)
        
        for img in train_images:
            dest = train_class / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        
        # Copy val images
        val_class = val_dir / class_name
        val_class.mkdir(exist_ok=True)
        
        for img in val_images:
            dest = val_class / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        
        total_train += len(train_images)
        total_val += len(val_images)
    
    # Save metadata
    metadata = {
        'num_classes': len(classes),
        'num_train': total_train,
        'num_val': total_val,
        'classes': sorted(classes),
        'plants': selected_plants
    }
    
    with open(target_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Filtered dataset created!")
    print(f"  Classes: {len(classes)}")
    print(f"  Train images: {total_train}")
    print(f"  Val images: {total_val}")
    print(f"  Location: {target_dir.absolute()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    filter_dataset()