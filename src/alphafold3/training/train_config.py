# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Training configuration for AlphaFold 3 fine-tuning."""

import dataclasses
from typing import Dict, Optional, Sequence


@dataclasses.dataclass
class LossWeights:
  """Weights for different loss components."""
  diffusion: float = 1.0
  plddt: float = 0.01
  pae: float = 0.1
  distogram: float = 0.01
  fape: float = 0.5


@dataclasses.dataclass
class OptimizationConfig:
  """Optimization hyperparameters."""
  learning_rate: float = 1e-4
  warmup_steps: int = 1000
  max_lr: float = 3e-4
  min_lr: float = 1e-6
  weight_decay: float = 1e-5
  gradient_clip_norm: float = 1.0
  adam_beta1: float = 0.9
  adam_beta2: float = 0.999
  adam_eps: float = 1e-8


@dataclasses.dataclass
class DiffusionTrainingConfig:
  """Configuration for diffusion model training."""
  # Noise schedule parameters
  sigma_min: float = 0.0004
  sigma_max: float = 160.0
  sigma_data: float = 16.0
  rho: float = 7.0  # Noise schedule parameter

  # Training steps
  num_diffusion_steps: int = 200  # Number of diffusion steps during training
  train_noise_schedule: str = 'log_linear'  # 'log_linear' or 'uniform'


@dataclasses.dataclass
class DataConfig:
  """Data loading configuration."""
  train_data_dir: str = ''
  val_data_dir: str = ''
  batch_size: int = 1  # Per-device batch size
  num_workers: int = 4
  shuffle: bool = True
  max_num_tokens: int = 512  # Maximum number of tokens per sample

  # Data augmentation
  use_augmentation: bool = True
  random_crop: bool = False
  crop_size: Optional[int] = None


@dataclasses.dataclass
class CheckpointConfig:
  """Checkpointing configuration."""
  checkpoint_dir: str = './checkpoints'
  save_interval_steps: int = 1000
  keep_n_checkpoints: int = 5
  resume_from_checkpoint: Optional[str] = None


@dataclasses.dataclass
class FineTuningConfig:
  """Fine-tuning specific configuration."""
  # Which parts of the model to train
  freeze_trunk: bool = False
  freeze_evoformer: bool = False
  freeze_diffusion_transformer: bool = False
  freeze_confidence_head: bool = False

  # Layers to unfreeze (if freezing is partial)
  unfreeze_last_n_layers: Optional[int] = None

  # Initial model parameters
  pretrained_model_dir: str = ''


@dataclasses.dataclass
class TrainingConfig:
  """Complete training configuration."""
  # Sub-configs
  optimization: OptimizationConfig = dataclasses.field(
      default_factory=OptimizationConfig
  )
  diffusion: DiffusionTrainingConfig = dataclasses.field(
      default_factory=DiffusionTrainingConfig
  )
  data: DataConfig = dataclasses.field(default_factory=DataConfig)
  checkpoint: CheckpointConfig = dataclasses.field(
      default_factory=CheckpointConfig
  )
  fine_tuning: FineTuningConfig = dataclasses.field(
      default_factory=FineTuningConfig
  )
  loss_weights: LossWeights = dataclasses.field(default_factory=LossWeights)

  # Training loop parameters
  num_train_steps: int = 100000
  log_interval_steps: int = 100
  eval_interval_steps: int = 1000
  num_eval_steps: int = 100

  # JAX/Device settings
  use_bfloat16: bool = True
  num_devices: int = 1

  # Random seed
  seed: int = 42

  # Experiment tracking
  experiment_name: str = 'af3_finetune'
  wandb_project: Optional[str] = None  # Set to enable W&B logging


def get_default_config() -> TrainingConfig:
  """Get default training configuration for fine-tuning."""
  return TrainingConfig()


def get_small_scale_config() -> TrainingConfig:
  """Get configuration for small-scale fine-tuning (e.g., on single GPU)."""
  config = TrainingConfig()
  config.data.batch_size = 1
  config.data.max_num_tokens = 256
  config.optimization.learning_rate = 5e-5
  config.num_train_steps = 10000
  config.checkpoint.save_interval_steps = 500
  return config


def get_adapter_config() -> TrainingConfig:
  """Get configuration for adapter-based fine-tuning.

  Freezes most of the model and only trains small adapter layers.
  This is memory-efficient and suitable for small datasets.
  """
  config = TrainingConfig()
  config.fine_tuning.freeze_trunk = True
  config.fine_tuning.freeze_evoformer = True
  config.fine_tuning.unfreeze_last_n_layers = 4
  config.optimization.learning_rate = 1e-4
  config.num_train_steps = 5000
  return config
