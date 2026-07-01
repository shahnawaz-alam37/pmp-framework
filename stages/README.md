# PMP-DACIS Stages

Split training pipeline into 4 independent stages for CPU temperature management.

## Files

| Stage | File | Description |
|-------|------|-------------|
| Stage 1 | `stage1_prune.py` | DACIS pruning (40%) + brief fine-tune |
| Stage 2 | `stage2_metalearn.py` | Meta-learning (ProtoNet) |
| Stage 3 | `stage3_refine.py` | Refinement pruning + final fine-tune |
| Stage 4 | `stage4_evaluate.py` | Evaluate 1/5/10-shot |

## Checkpoint Flow

```
checkpoint/stage1_model.pth → checkpoint/stage2_model.pth → checkpoint/stage3_model.pth
```

## Running

### Run all stages at once:
```bash
python run_all_stages.py
```

### Run stages individually:
```bash
python stages/stage1_prune.py
python stages/stage2_metalearn.py
python stages/stage3_refine.py
python stages/stage4_evaluate.py
```

## Output

- Checkpoints saved to `checkpoints/` directory
- Evaluation results saved to `results/final_evaluation.json`