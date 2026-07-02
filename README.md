# PMP-DACIS: Plant Disease Detection via Pruned Meta-Learning
[![arXiv](https://img.shields.io/badge/arXiv-2601.02353-b31b1b.svg)](https://doi.org/10.48550/arXiv.2601.02353) <br>
**PMP-DACIS** (Pruned Meta-Learning with Disease-Aware Channel Importance Scoring) is a few-shot learning framework for plant disease classification. It uses a **ResNet-18** backbone (pre-trained on ImageNet via `torchvision.models`) and applies structured channel pruning with DACIS scoring to create compact models that can classify plant diseases from just 1-5 examples per class.

## Model Architecture

- **Backbone**: `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` — a standard ResNet-18 provided by PyTorch, pre-trained on ImageNet-1K.
- **Pruning**: DACIS (Disease-Aware Channel Importance Scoring) — combines gradient sensitivity, activation variance, and Fisher discriminant ratio.
- **Few-Shot Learning**: Prototypical Networks (ProtoNet) — classifies query images by computing distances to class prototypes from support examples.

## Project Structure

```
├── config.py                  # Training/pruning configuration
├── dataset.py                 # Dataset & few-shot data loader
├── filter_dataset.py          # Filter PlantVillage → selected plants
├── main.py                    # End-to-end training (full + few-shot)
├── stages/
│   ├── stage1_prune.py        # Stage 1: DACIS pruning + fine-tune
│   ├── stage2_metalearn.py    # Stage 2: ProtoNet meta-learning
│   ├── stage3_refine.py       # Stage 3: Refinement pruning + fine-tune
│   ├── stage4_evaluate.py     # Stage 4: 1/5/10-shot evaluation
│   └── test_inference.py      # Quick inference helper
├── models/
│   ├── backbone.py            # ResNet-18 backbone + prunable variant
│   ├── prototypical.py        # Prototypical Network implementation
│   ├── dacis.py               # DACIS scoring + channel pruner
│   └── maml.py                # MAML implementation (optional)
└── requirements.txt           # Python dependencies
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-compatible GPU for faster training

### 2. Clone & Install

```bash
git clone https://github.com/shahnawaz-alam37/pmp-framework.git
cd pmp-framework
pip install -r requirements.txt
```

### 3. Download the PlantVillage Dataset

1. Download the **PlantVillage** dataset from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (or the [official source](https://github.com/spMohanty/PlantVillage-Dataset)).
2. Extract the dataset so it has the following structure:
   ```
   data/
   ├── train/
   │   ├── Apple___Apple_scab/
   │   ├── Apple___Black_rot/
   │   ├── ... (one folder per disease class, containing .jpg images)
   └── val/
       ├── Apple___Apple_scab/
       ├── ...
   ```

### 4. Filter the Dataset (Optional)

To train only on the **8 selected plant types** (Tomato, Potato, Pepper, Strawberry, Corn, Apple, Cherry, Grape) instead of the full PlantVillage dataset:

```bash
python filter_dataset.py
```

**To configure `filter_dataset.py`:**
- Edit the `selected_plants` list at line 19 to include/exclude plant types.
- Change `source_dir` (line 15) to point to your raw PlantVillage data.
- Change `target_dir` (line 16) to the output directory (default: `data_filtered/`).
- The script copies 80% of images for training and 20% for validation.

This creates a `data_filtered/` directory with `train/` and `val/` subfolders.

### 5. Run the Full Pipeline (4 Stages)

Run all stages sequentially:

```bash
python stages/stage1_prune.py     # DACIS pruning + fine-tuning
python stages/stage2_metalearn.py # ProtoNet meta-learning
python stages/stage3_refine.py    # Refinement pruning + fine-tuning
python stages/stage4_evaluate.py  # 1/5/10-shot evaluation
```

Or run each stage individually (recommended for monitoring). Checkpoints are saved to `checkpoints/` after each stage:
- `checkpoints/stage1_model.pth` → `checkpoints/stage2_model.pth` → `checkpoints/stage3_model.pth`

### 6. Run End-to-End Training (Alternative)

```bash
python main.py
```

This runs full supervised pre-training followed by few-shot meta-learning in a single script.

### 7. View Results

Evaluation results are saved to `results/final_evaluation.json`.

## Configuration (`config.py`)

| Key | Default | Description |
|-----|---------|-------------|
| `dataset.data_dir` | `data_filtered` | Path to dataset |
| `dataset.num_classes` | `41` | Number of disease classes |
| `pipeline.stage1_prune_ratio` | `0.35` | Pruning ratio for Stage 1 |
| `pipeline.stage2_epochs` | `20` | Meta-learning epochs |
| `pipeline.total_compression` | `0.65` | Target total compression |
| `fewshot.n_way` | `5` | Classes per episode |
| `fewshot.k_shot` | `1-10` | Support examples per class |

## Notes

- The ResNet-18 model is **not a custom model** — it comes directly from PyTorch's `torchvision.models` library, pre-trained on ImageNet-1K (`IMAGENET1K_V1` weights). The code initializes it with `models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` and replaces the final fully-connected layer for plant disease classification.
- Training on CPU is possible but slow. A GPU with 4GB+ VRAM is recommended.
- The 4-stage pipeline was designed to manage temperature on resource-constrained hardware.

## License

MIT
