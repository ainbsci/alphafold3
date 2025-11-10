# AlphaFold 3 Training and Fine-tuning

This document explains how to fine-tune AlphaFold 3 on custom datasets.

⚠️ **IMPORTANT DISCLAIMER** ⚠️

This training code is a **reference implementation** created to enable fine-tuning on custom data. It is:

- ✅ Based on the AlphaFold 3 paper and codebase architecture
- ✅ Suitable for fine-tuning on domain-specific datasets
- ⚠️ **NOT** the original DeepMind training code
- ⚠️ **NOT** validated on large-scale training runs
- ⚠️ Requires significant computational resources (GPUs)

**Use at your own risk and validate results carefully.**

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Data Preparation](#data-preparation)
4. [Quick Start](#quick-start)
5. [Training Modes](#training-modes)
6. [Configuration](#configuration)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The training framework implements:

### Loss Functions
- **Diffusion Loss**: Main loss for structure prediction (EDM-style)
- **pLDDT Loss**: Per-residue confidence prediction
- **PAE Loss**: Predicted aligned error
- **Distogram Loss**: Distance distribution prediction
- **FAPE Loss**: Frame-aligned point error

### Features
- ✅ Fine-tuning from pretrained AlphaFold 3 parameters
- ✅ Selective layer freezing
- ✅ Learning rate scheduling with warmup
- ✅ Gradient clipping and AdamW optimization
- ✅ Checkpoint saving/loading
- ✅ Validation evaluation
- ✅ Weights & Biases integration

---

## Requirements

### Hardware
- **Minimum**: 1x NVIDIA A100 (80GB) or equivalent
- **Recommended**: Multiple A100/H100 GPUs
- **RAM**: 128GB+ system RAM
- **Storage**: 500GB+ for datasets and checkpoints

### Software
- Python 3.11+
- JAX with CUDA 12 support
- All AlphaFold 3 dependencies (see `requirements.txt`)
- Optional: `wandb` for experiment tracking

### Installation

```bash
# Install AlphaFold 3 dependencies
pip install -r requirements.txt

# Optional: Install Weights & Biases
pip install wandb
```

---

## Data Preparation

### Directory Structure

Organize your training data as follows:

```
my_training_data/
├── structures/          # Ground truth structures
│   ├── protein1.cif    # mmCIF format
│   ├── protein2.cif
│   └── ...
└── inputs/             # Optional: AlphaFold3 JSON inputs
    ├── protein1.json
    ├── protein2.json
    └── ...
```

### Data Formats

#### Option 1: mmCIF Files Only

Place ground truth structures in `structures/`:

```
structures/
├── 1abc.cif
├── 2def.cif
└── 3ghi.cif
```

The data loader will automatically extract sequences and create inputs.

#### Option 2: JSON Inputs + Structures

For more control, provide both JSON inputs and structures:

```json
{
  "name": "my_protein",
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "MKLAVLALALAVLALA..."
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

### Data Requirements

- **Minimum dataset size**: 100+ structures for meaningful fine-tuning
- **Recommended**: 1,000+ structures
- **Maximum tokens**: Default 512 (configurable via `--max_num_tokens`)
- **Quality**: High-resolution structures (< 3Å) recommended

---

## Quick Start

### 1. Basic Fine-tuning

```bash
python train_alphafold.py \
  --train_data_dir=/path/to/training/data \
  --model_dir=/path/to/pretrained/alphafold3/models \
  --output_dir=/path/to/output \
  --num_train_steps=10000 \
  --learning_rate=1e-4
```

### 2. Resume from Checkpoint

```bash
python train_alphafold.py \
  --train_data_dir=/path/to/training/data \
  --output_dir=/path/to/output \
  --resume_from_checkpoint=/path/to/checkpoint.pkl
```

### 3. With Validation

```bash
python train_alphafold.py \
  --train_data_dir=/path/to/training/data \
  --val_data_dir=/path/to/validation/data \
  --model_dir=/path/to/pretrained/model \
  --output_dir=/path/to/output \
  --num_train_steps=10000 \
  --eval_interval=500
```

---

## Training Modes

### Mode 1: Full Fine-tuning (Default)

Train all parameters:

```bash
python train_alphafold.py \
  --train_data_dir=./data/train \
  --model_dir=./models \
  --output_dir=./output_full
```

**Use when**: You have a large dataset (10,000+ structures) and lots of compute.

### Mode 2: Frozen Trunk

Freeze the Evoformer trunk, train only the diffusion head:

```bash
python train_alphafold.py \
  --train_data_dir=./data/train \
  --model_dir=./models \
  --output_dir=./output_frozen_trunk \
  --freeze_trunk
```

**Use when**: Smaller dataset, want to preserve pretrained features.

### Mode 3: Frozen Diffusion

Freeze diffusion transformer, train only trunk and confidence heads:

```bash
python train_alphafold.py \
  --train_data_dir=./data/train \
  --model_dir=./models \
  --output_dir=./output_frozen_diffusion \
  --freeze_diffusion
```

**Use when**: Fine-tuning for confidence prediction or MSA-dependent features.

### Mode 4: Minimal Fine-tuning

Freeze most layers, train only last few:

```python
# In custom script using train_config
config = train_config.get_adapter_config()
config.data.train_data_dir = './data/train'
config.fine_tuning.pretrained_model_dir = './models'
```

**Use when**: Very small dataset (< 100 structures), want to avoid overfitting.

---

## Configuration

### Command-line Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--train_data_dir` | - | **Required**. Training data directory |
| `--output_dir` | - | **Required**. Output directory |
| `--model_dir` | - | Pretrained model directory |
| `--val_data_dir` | None | Validation data directory |
| `--num_train_steps` | 10000 | Total training steps |
| `--batch_size` | 1 | Batch size (only 1 supported) |
| `--learning_rate` | 1e-4 | Peak learning rate |
| `--warmup_steps` | 1000 | Warmup steps |
| `--freeze_trunk` | False | Freeze Evoformer trunk |
| `--freeze_diffusion` | False | Freeze diffusion transformer |
| `--checkpoint_interval` | 1000 | Checkpoint save interval |
| `--log_interval` | 100 | Logging interval |
| `--eval_interval` | 1000 | Evaluation interval |
| `--gpu_device` | 0 | GPU device index |
| `--wandb_project` | None | W&B project name |

### Loss Weights

Default loss weights (can be modified in `train_config.py`):

```python
LossWeights(
    diffusion=1.0,    # Main structure loss
    plddt=0.01,       # Confidence loss
    pae=0.1,          # Alignment error loss
    distogram=0.01,   # Distance prediction
    fape=0.5,         # Frame-aligned error
)
```

### Learning Rate Schedule

Default schedule:
- **Warmup**: 1,000 steps (linear from 0 to max LR)
- **Peak LR**: 1e-4 (configurable)
- **Decay**: Cosine decay to min LR (1e-6)

---

## Advanced Usage

### Custom Configuration

Create a custom config:

```python
from alphafold3.training import train_config

# Create custom config
config = train_config.TrainingConfig()

# Modify settings
config.optimization.learning_rate = 5e-5
config.optimization.warmup_steps = 2000
config.loss_weights.diffusion = 2.0
config.loss_weights.pae = 0.2

# Use in training script
# (See train_alphafold.py for integration)
```

### Multi-GPU Training

Currently single-GPU only. Multi-GPU support requires data parallelism:

```python
# Future implementation
# Use jax.pmap for data parallelism
# Replicate model across devices
# Split batch across devices
```

### Custom Loss Functions

Add custom losses in `src/alphafold3/training/losses.py`:

```python
def my_custom_loss(predicted, target, mask):
    """Your custom loss."""
    error = jnp.square(predicted - target)
    return jnp.sum(error * mask) / jnp.sum(mask)
```

Then modify `compute_total_loss()` to include it.

### Weights & Biases Integration

```bash
# Install wandb
pip install wandb
wandb login

# Run with W&B logging
python train_alphafold.py \
  --train_data_dir=./data/train \
  --model_dir=./models \
  --output_dir=./output \
  --wandb_project=my-af3-finetuning \
  --experiment_name=exp_001
```

View metrics at https://wandb.ai/your-username/my-af3-finetuning

---

## Monitoring Training

### Console Output

```
Step 100/10000 (2.45s/step) - total_loss: 145.2341, grad_norm: 1.2345, loss/diffusion: 142.1, loss/plddt: 0.234
Step 200/10000 (2.43s/step) - total_loss: 132.4567, grad_norm: 1.1234, loss/diffusion: 130.2, loss/plddt: 0.198
...
Saved checkpoint to ./output/checkpoints/checkpoint_1000.pkl
```

### Key Metrics to Watch

1. **total_loss**: Should decrease steadily
2. **loss/diffusion**: Main structure prediction loss (largest component)
3. **grad_norm**: Should stay < 10 (indicates stable training)
4. **eval metrics**: Validation loss should not increase (overfitting check)

### Expected Training Time

| Dataset Size | Steps | GPU | Time |
|--------------|-------|-----|------|
| 100 structures | 1,000 | A100 | ~1 hour |
| 1,000 structures | 10,000 | A100 | ~7 hours |
| 10,000 structures | 100,000 | A100 | ~3 days |

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA OOM error

**Solutions**:
```bash
# Reduce max tokens
python train_alphafold.py --max_num_tokens=256 ...

# Enable unified memory (in Dockerfile)
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV TF_FORCE_UNIFIED_MEMORY=true
ENV XLA_CLIENT_MEM_FRACTION=3.2
```

### Loss Not Decreasing

**Possible causes**:
1. Learning rate too high → Try `--learning_rate=5e-5`
2. Too many frozen layers → Unfreeze more layers
3. Bad data quality → Check ground truth structures
4. Gradient explosion → Check `grad_norm` values

### Nan/Inf in Losses

**Solutions**:
```python
# In train_config.py, reduce learning rate
config.optimization.learning_rate = 1e-5
config.optimization.gradient_clip_norm = 0.5  # More aggressive clipping
```

### Slow Training

**Solutions**:
1. Check data loading (not causing bottleneck)
2. Use JAX compilation cache: `--jax_compilation_cache_dir=./cache`
3. Ensure using GPU: Check `jax.local_devices()`
4. Profile with `jax.profiler`

### Data Loading Errors

**Error**: `No structure or input files found`

**Solution**: Check directory structure:
```bash
ls -R my_training_data/
# Should show:
# structures/
# structures/file1.cif
# structures/file2.cif
```

---

## Validation and Testing

### After Fine-tuning

1. **Run inference** on test structures:
```bash
python run_alphafold.py \
  --json_path=./test_input.json \
  --model_dir=./output/checkpoints/checkpoint_10000 \
  --output_dir=./test_output \
  --norun_data_pipeline
```

2. **Compare metrics**:
   - Compare pLDDT scores
   - Check PAE distributions
   - Visual inspection in PyMOL/ChimeraX

3. **Benchmark** against baseline:
   - Run same test set with original AF3
   - Compare RMSD, TM-score, lDDT

---

## Best Practices

### 1. Start Small
- Begin with 100-1000 structures
- Use frozen trunk mode
- Train for 1,000-5,000 steps
- Validate on held-out test set

### 2. Data Quality Matters
- Use high-resolution structures (< 2.5Å preferred)
- Check for missing residues
- Verify sequence matches structure
- Remove low-quality/problematic structures

### 3. Monitor Carefully
- Watch for overfitting (eval loss increasing)
- Check gradient norms (should be < 5)
- Save checkpoints frequently
- Keep validation set separate

### 4. Hyperparameter Tuning
- Learning rate: Try [1e-5, 5e-5, 1e-4]
- Warmup steps: 10-20% of total steps
- Loss weights: Adjust based on which metric you care about
- Freezing: More frozen layers = less overfitting

---

## Citation

If you use this training code, please cite both the original AlphaFold 3 paper and acknowledge this implementation:

```bibtex
@article{Abramson2024,
  author = {Abramson, Josh and Adler, Jonas and Dunger, Jack and ...},
  title = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  journal = {Nature},
  year = {2024},
  doi = {10.1038/s41586-024-07487-w}
}
```

---

## Support and Contribution

This is a community implementation. For issues:
1. Check [troubleshooting](#troubleshooting) section
2. Review AlphaFold 3 paper for algorithm details
3. Open an issue on GitHub with:
   - Error message
   - Command used
   - System info (GPU, JAX version)
   - Training logs

**Note**: This is not officially supported by DeepMind.

---

## License

This training code follows the same license as AlphaFold 3:
- **Source code**: CC BY-NC-SA 4.0
- **Model parameters**: Subject to AlphaFold 3 terms of use
- **For non-commercial use only**

See `LICENSE` and `WEIGHTS_TERMS_OF_USE.md` for details.
