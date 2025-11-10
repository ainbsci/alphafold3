# AlphaFold 3 Training Module

This module provides training and fine-tuning capabilities for AlphaFold 3.

## ⚠️ Important Disclaimer

This is a **community-created training implementation**, not the original DeepMind training code. It is:

- Based on the AlphaFold 3 paper methodology
- Designed for fine-tuning on custom datasets
- **Not validated on full-scale training from scratch**
- Provided as-is for research purposes

## Module Structure

```
training/
├── __init__.py           # Module exports
├── losses.py             # Loss functions (diffusion, pLDDT, PAE, etc.)
├── trainer.py            # Training loop and optimization
├── train_config.py       # Configuration dataclasses
├── data_loader.py        # Dataset and data loading utilities
└── README.md             # This file
```

## Quick Start

See the main [TRAINING.md](../../../TRAINING.md) documentation for complete usage instructions.

### Minimal Example

```python
from alphafold3.training import trainer, train_config, data_loader
from alphafold3.model import model
import pathlib

# Create configurations
model_cfg = model.Model.Config()
train_cfg = train_config.get_small_scale_config()
train_cfg.data.train_data_dir = './data/train'
train_cfg.fine_tuning.pretrained_model_dir = './models'

# Create trainer
af3_trainer = trainer.AlphaFold3Trainer(
    model_config=model_cfg,
    train_config=train_cfg,
    pretrained_params_dir=pathlib.Path('./models')
)

# Load dataset
dataset = data_loader.StructureDataset(
    data_dir='./data/train',
    max_num_tokens=256
)

# Load pretrained parameters
params = af3_trainer.load_pretrained_params()
opt_state = af3_trainer.optimizer.init(params)

# Training loop (simplified)
for step in range(1000):
    example = dataset[step % len(dataset)]
    params, opt_state, metrics = af3_trainer.train_step(
        params, opt_state, example['batch'],
        example['true_positions'], rng_key
    )
```

## Components

### 1. Loss Functions (`losses.py`)

Implements all training losses:

- **`diffusion_loss()`**: Main structure prediction loss using EDM framework
- **`plddt_loss()`**: Per-residue confidence (local distance difference test)
- **`pae_loss()`**: Predicted aligned error
- **`distogram_loss()`**: Distance histogram prediction
- **`fape_loss()`**: Frame-aligned point error
- **`compute_total_loss()`**: Combines all losses with configurable weights

### 2. Trainer (`trainer.py`)

Main training orchestration:

- **`AlphaFold3Trainer`**: Main trainer class
  - `train_step()`: Single training step with gradient update
  - `eval_step()`: Evaluation without gradient computation
  - `save_checkpoint()`: Save model and optimizer state
  - `load_checkpoint()`: Resume from checkpoint
  - `load_pretrained_params()`: Load AlphaFold 3 pretrained weights

Features:
- AdamW optimizer with cosine learning rate schedule
- Gradient clipping
- Selective layer freezing
- JAX JIT compilation for performance

### 3. Configuration (`train_config.py`)

Dataclass-based configuration system:

- **`TrainingConfig`**: Master configuration
- **`OptimizationConfig`**: Learning rate, warmup, gradient clipping
- **`DiffusionTrainingConfig`**: Diffusion-specific parameters
- **`DataConfig`**: Dataset and batching settings
- **`CheckpointConfig`**: Checkpoint management
- **`FineTuningConfig`**: Layer freezing options
- **`LossWeights`**: Loss component weights

Preset configurations:
- `get_default_config()`: Standard fine-tuning
- `get_small_scale_config()`: Single GPU, small dataset
- `get_adapter_config()`: Minimal parameter tuning

### 4. Data Loading (`data_loader.py`)

Dataset and data pipeline:

- **`StructureDataset`**: Main dataset class
  - Loads from mmCIF files
  - Loads from AlphaFold3 JSON inputs
  - Caching for performance
  - Automatic featurization

- **`create_dataloader()`**: Iterator for training
- **`collate_fn()`**: Batch collation (currently batch_size=1)

## Usage Patterns

### Pattern 1: Full Fine-tuning

```python
config = train_config.TrainingConfig()
config.fine_tuning.freeze_trunk = False
config.fine_tuning.freeze_diffusion_transformer = False
# Train all parameters
```

### Pattern 2: Frozen Trunk

```python
config = train_config.TrainingConfig()
config.fine_tuning.freeze_trunk = True
# Only train diffusion head and confidence heads
```

### Pattern 3: Adapter-style

```python
config = train_config.get_adapter_config()
# Freeze most layers, train only last few
config.fine_tuning.unfreeze_last_n_layers = 4
```

## Advanced Topics

### Custom Loss Weights

```python
config = train_config.TrainingConfig()
config.loss_weights.diffusion = 2.0  # Emphasize structure accuracy
config.loss_weights.pae = 0.05       # De-emphasize alignment error
```

### Learning Rate Tuning

```python
config = train_config.TrainingConfig()
config.optimization.max_lr = 5e-5
config.optimization.min_lr = 1e-7
config.optimization.warmup_steps = 2000
```

### Custom Diffusion Schedule

```python
config = train_config.TrainingConfig()
config.diffusion.sigma_min = 0.001
config.diffusion.sigma_max = 200.0
config.diffusion.num_diffusion_steps = 250
```

## Loss Function Details

### Diffusion Loss

Based on Karras et al. EDM framework:

```
loss = λ(σ) * || predicted_pos - true_pos ||²
where λ(σ) = 1 / (σ² + σ_data²)
```

### pLDDT Loss

Cross-entropy over discretized LDDT bins:

```
loss = -log P(lddt_bin | predicted_logits)
where lddt_bin ∈ {0, 1, ..., 49}
```

### PAE Loss

Position error when aligned on different residues:

```
loss = -log P(error_bin | predicted_logits)
where error_bin ∈ {0, 1, ..., 63}
```

## Performance Considerations

### Memory Usage

- **Batch size 1**: ~50-70GB GPU memory (512 tokens)
- **Batch size 1**: ~30-40GB GPU memory (256 tokens)
- Larger sequences require more memory quadratically

### Training Speed

- **A100 80GB**: ~2-3 seconds/step (512 tokens)
- **H100 80GB**: ~1-2 seconds/step (512 tokens)
- Compilation: First step is slow (~30s), subsequent steps are fast

### Optimization Tips

1. **Use JAX compilation cache**: Set `--jax_compilation_cache_dir`
2. **Reduce max tokens**: Use `--max_num_tokens=256` for faster training
3. **Freeze layers**: Less parameters = faster + less memory
4. **Profile**: Use `jax.profiler` to find bottlenecks

## Testing

Currently no automated tests. To test manually:

```bash
# Test data loading
python -c "
from alphafold3.training.data_loader import StructureDataset
dataset = StructureDataset('./test_data')
print(f'Loaded {len(dataset)} examples')
"

# Test loss computation
python -c "
from alphafold3.training import losses
import jax.numpy as jnp
loss = losses.diffusion_loss(
    jnp.ones((10, 20, 3)),
    jnp.zeros((10, 20, 3)),
    jnp.ones((10, 20)),
    1.0
)
print(f'Loss: {loss}')
"
```

## Known Limitations

1. **Batch size**: Currently only batch_size=1 supported (variable sequence lengths)
2. **Multi-GPU**: Not yet implemented (requires pmap/data parallelism)
3. **Validation**: No automated validation for numerical correctness
4. **Data augmentation**: Minimal (only random rigid augmentation)

## Contributing

To add new features:

1. **New loss**: Add to `losses.py` and update `compute_total_loss()`
2. **New optimizer**: Modify `_init_optimizer()` in `trainer.py`
3. **New data source**: Extend `StructureDataset` in `data_loader.py`
4. **New config**: Add fields to relevant config in `train_config.py`

## References

- AlphaFold 3 paper: https://doi.org/10.1038/s41586-024-07487-w
- EDM paper (Karras et al.): https://arxiv.org/abs/2206.00364
- JAX documentation: https://jax.readthedocs.io/
- Haiku documentation: https://dm-haiku.readthedocs.io/

## Support

This is an unofficial implementation. For issues:

1. Check [TRAINING.md](../../../TRAINING.md) troubleshooting section
2. Review AlphaFold 3 paper for algorithm details
3. Open GitHub issue with error details

**Not supported by DeepMind/Google.**
