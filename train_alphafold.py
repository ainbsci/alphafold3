# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Training/Fine-tuning script for AlphaFold 3.

This script enables fine-tuning AlphaFold 3 on custom datasets.

Example usage:
  # Fine-tune on custom dataset
  python train_alphafold.py \\
    --train_data_dir=/path/to/training/data \\
    --model_dir=/path/to/pretrained/model \\
    --output_dir=/path/to/output \\
    --num_train_steps=10000

  # Resume from checkpoint
  python train_alphafold.py \\
    --train_data_dir=/path/to/training/data \\
    --resume_from_checkpoint=/path/to/checkpoint.pkl \\
    --output_dir=/path/to/output
"""

import os
import pathlib
import time
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp

from alphafold3.model import model as af3_model
from alphafold3.training import train_config
from alphafold3.training import trainer
from alphafold3.training import data_loader


# Data paths
_TRAIN_DATA_DIR = flags.DEFINE_string(
    'train_data_dir',
    None,
    'Path to training data directory. Should contain structures/ or inputs/ subdirs.',
)
_VAL_DATA_DIR = flags.DEFINE_string(
    'val_data_dir',
    None,
    'Path to validation data directory (optional).',
)
_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    None,
    'Path to pretrained AlphaFold 3 model parameters.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to output directory for checkpoints and logs.',
)

# Training parameters
_NUM_TRAIN_STEPS = flags.DEFINE_integer(
    'num_train_steps',
    10000,
    'Number of training steps.',
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    1,
    'Batch size (currently only 1 is supported).',
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    1e-4,
    'Peak learning rate.',
)
_WARMUP_STEPS = flags.DEFINE_integer(
    'warmup_steps',
    1000,
    'Number of warmup steps.',
)

# Fine-tuning options
_FREEZE_TRUNK = flags.DEFINE_bool(
    'freeze_trunk',
    False,
    'Whether to freeze the trunk (evoformer) during fine-tuning.',
)
_FREEZE_DIFFUSION = flags.DEFINE_bool(
    'freeze_diffusion',
    False,
    'Whether to freeze the diffusion transformer during fine-tuning.',
)

# Checkpointing
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'checkpoint_interval',
    1000,
    'Save checkpoint every N steps.',
)
_RESUME_FROM_CHECKPOINT = flags.DEFINE_string(
    'resume_from_checkpoint',
    None,
    'Path to checkpoint to resume from.',
)

# Logging
_LOG_INTERVAL = flags.DEFINE_integer(
    'log_interval',
    100,
    'Log metrics every N steps.',
)
_EVAL_INTERVAL = flags.DEFINE_integer(
    'eval_interval',
    1000,
    'Evaluate on validation set every N steps.',
)

# JAX configuration
_GPU_DEVICE = flags.DEFINE_integer(
    'gpu_device',
    0,
    'GPU device to use.',
)

# Experiment tracking
_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name',
    'af3_finetune',
    'Name of the experiment.',
)
_WANDB_PROJECT = flags.DEFINE_string(
    'wandb_project',
    None,
    'Weights & Biases project name (optional).',
)


def setup_logging(output_dir: pathlib.Path):
  """Setup logging to file and console."""
  log_file = output_dir / 'training.log'
  logging.get_absl_handler().use_absl_log_file('training', str(output_dir))
  logging.info(f'Logging to {log_file}')


def create_model_config() -> af3_model.Model.Config:
  """Create model configuration for training."""
  # Use default config from inference
  config = af3_model.Model.Config()

  # Training-specific modifications
  config.heads.diffusion.eval.num_samples = 1  # Single sample during training
  config.return_embeddings = False  # Don't need embeddings during training

  return config


def create_training_config() -> train_config.TrainingConfig:
  """Create training configuration from flags."""
  config = train_config.TrainingConfig()

  # Data
  config.data.train_data_dir = _TRAIN_DATA_DIR.value
  config.data.val_data_dir = _VAL_DATA_DIR.value or ''
  config.data.batch_size = _BATCH_SIZE.value

  # Optimization
  config.optimization.learning_rate = _LEARNING_RATE.value
  config.optimization.max_lr = _LEARNING_RATE.value
  config.optimization.warmup_steps = _WARMUP_STEPS.value

  # Training loop
  config.num_train_steps = _NUM_TRAIN_STEPS.value
  config.log_interval_steps = _LOG_INTERVAL.value
  config.eval_interval_steps = _EVAL_INTERVAL.value

  # Fine-tuning
  config.fine_tuning.pretrained_model_dir = _MODEL_DIR.value
  config.fine_tuning.freeze_trunk = _FREEZE_TRUNK.value
  config.fine_tuning.freeze_diffusion_transformer = _FREEZE_DIFFUSION.value

  # Checkpointing
  config.checkpoint.checkpoint_dir = _OUTPUT_DIR.value
  config.checkpoint.save_interval_steps = _CHECKPOINT_INTERVAL.value
  config.checkpoint.resume_from_checkpoint = _RESUME_FROM_CHECKPOINT.value

  # Experiment
  config.experiment_name = _EXPERIMENT_NAME.value
  config.wandb_project = _WANDB_PROJECT.value

  return config


def train_loop(
    af3_trainer: trainer.AlphaFold3Trainer,
    train_dataset: data_loader.StructureDataset,
    val_dataset: Optional[data_loader.StructureDataset],
    config: train_config.TrainingConfig,
    output_dir: pathlib.Path,
):
  """Main training loop.

  Args:
    af3_trainer: Trainer instance
    train_dataset: Training dataset
    val_dataset: Validation dataset (optional)
    config: Training configuration
    output_dir: Output directory
  """
  # Initialize parameters and optimizer state
  if config.checkpoint.resume_from_checkpoint:
    logging.info(f'Resuming from {config.checkpoint.resume_from_checkpoint}')
    params, opt_state, start_step = af3_trainer.load_checkpoint(
        pathlib.Path(config.checkpoint.resume_from_checkpoint)
    )
  else:
    logging.info('Initializing from pretrained parameters')
    params = af3_trainer.load_pretrained_params()

    # Initialize optimizer state
    opt_state = af3_trainer.optimizer.init(params)
    start_step = 0

  # Create RNG key
  rng = jax.random.PRNGKey(config.seed)

  # Training loop
  logging.info(f'Starting training from step {start_step}')

  for step in range(start_step, config.num_train_steps):
    step_start_time = time.time()

    # Get training batch
    # In practice, you'd iterate through the dataset more efficiently
    train_example = train_dataset[step % len(train_dataset)]

    # Split RNG
    rng, step_rng = jax.random.split(rng)

    # Training step
    params, opt_state, metrics = af3_trainer.train_step(
        params=params,
        opt_state=opt_state,
        batch=train_example['batch'],
        true_positions=train_example['true_positions'],
        rng_key=step_rng,
    )

    step_time = time.time() - step_start_time

    # Logging
    if step % config.log_interval_steps == 0:
      metrics_str = ', '.join(
          f'{k}: {float(v):.4f}' for k, v in metrics.items()
      )
      logging.info(
          f'Step {step}/{config.num_train_steps} '
          f'({step_time:.2f}s/step) - {metrics_str}'
      )

      # Log to W&B if enabled
      if config.wandb_project:
        try:
          import wandb
          wandb.log(
              {**{k: float(v) for k, v in metrics.items()}, 'step': step}
          )
        except ImportError:
          pass

    # Evaluation
    if val_dataset and step % config.eval_interval_steps == 0 and step > 0:
      logging.info(f'Running evaluation at step {step}')
      eval_metrics = run_evaluation(
          af3_trainer, val_dataset, params, rng
      )
      eval_str = ', '.join(
          f'{k}: {float(v):.4f}' for k, v in eval_metrics.items()
      )
      logging.info(f'Eval metrics: {eval_str}')

      if config.wandb_project:
        try:
          import wandb
          wandb.log(
              {f'eval/{k}': float(v) for k, v in eval_metrics.items()}
          )
        except ImportError:
          pass

    # Save checkpoint
    if step % config.checkpoint.save_interval_steps == 0 and step > 0:
      af3_trainer.save_checkpoint(
          params=params,
          opt_state=opt_state,
          step=step,
          checkpoint_dir=output_dir / 'checkpoints',
      )

  # Final checkpoint
  logging.info('Training complete. Saving final checkpoint.')
  af3_trainer.save_checkpoint(
      params=params,
      opt_state=opt_state,
      step=config.num_train_steps,
      checkpoint_dir=output_dir / 'checkpoints',
  )


def run_evaluation(
    af3_trainer: trainer.AlphaFold3Trainer,
    val_dataset: data_loader.StructureDataset,
    params,
    rng,
) -> dict:
  """Run evaluation on validation set.

  Args:
    af3_trainer: Trainer instance
    val_dataset: Validation dataset
    params: Model parameters
    rng: Random key

  Returns:
    Dictionary of averaged validation metrics
  """
  all_metrics = []
  num_eval_samples = min(100, len(val_dataset))

  for i in range(num_eval_samples):
    example = val_dataset[i]
    rng, step_rng = jax.random.split(rng)

    metrics = af3_trainer.eval_step(
        params=params,
        batch=example['batch'],
        true_positions=example['true_positions'],
        rng_key=step_rng,
    )
    all_metrics.append(metrics)

  # Average metrics
  avg_metrics = {
      k: jnp.mean(jnp.array([m[k] for m in all_metrics]))
      for k in all_metrics[0].keys()
  }

  return avg_metrics


def main(_):
  """Main function."""
  # Validate flags
  if _TRAIN_DATA_DIR.value is None:
    raise ValueError('--train_data_dir must be specified')
  if _OUTPUT_DIR.value is None:
    raise ValueError('--output_dir must be specified')

  # Create output directory
  output_dir = pathlib.Path(_OUTPUT_DIR.value)
  output_dir.mkdir(parents=True, exist_ok=True)

  # Setup logging
  setup_logging(output_dir)

  logging.info('='*80)
  logging.info('AlphaFold 3 Fine-tuning')
  logging.info('='*80)

  # Setup JAX
  devices = jax.local_devices(backend='gpu')
  if devices:
    device = devices[_GPU_DEVICE.value]
    logging.info(f'Using device: {device}')
  else:
    logging.warning('No GPU found, using CPU')

  # Create configurations
  model_config = create_model_config()
  training_config = create_training_config()

  # Log configuration
  logging.info('Training configuration:')
  logging.info(f'  Train data: {training_config.data.train_data_dir}')
  logging.info(f'  Model dir: {training_config.fine_tuning.pretrained_model_dir}')
  logging.info(f'  Output dir: {output_dir}')
  logging.info(f'  Steps: {training_config.num_train_steps}')
  logging.info(f'  Learning rate: {training_config.optimization.learning_rate}')
  logging.info(f'  Freeze trunk: {training_config.fine_tuning.freeze_trunk}')

  # Initialize W&B if requested
  if training_config.wandb_project:
    try:
      import wandb
      wandb.init(
          project=training_config.wandb_project,
          name=training_config.experiment_name,
          config=training_config.__dict__,
      )
      logging.info(f'Initialized W&B project: {training_config.wandb_project}')
    except ImportError:
      logging.warning('wandb not installed, skipping W&B logging')

  # Load datasets
  logging.info('Loading training dataset...')
  train_dataset = data_loader.StructureDataset(
      data_dir=training_config.data.train_data_dir,
      max_num_tokens=training_config.data.max_num_tokens,
  )
  logging.info(f'Training dataset size: {len(train_dataset)}')

  val_dataset = None
  if training_config.data.val_data_dir:
    logging.info('Loading validation dataset...')
    val_dataset = data_loader.StructureDataset(
        data_dir=training_config.data.val_data_dir,
        max_num_tokens=training_config.data.max_num_tokens,
    )
    logging.info(f'Validation dataset size: {len(val_dataset)}')

  # Create trainer
  logging.info('Initializing trainer...')
  af3_trainer = trainer.AlphaFold3Trainer(
      model_config=model_config,
      train_config=training_config,
      pretrained_params_dir=pathlib.Path(
          training_config.fine_tuning.pretrained_model_dir
      )
      if training_config.fine_tuning.pretrained_model_dir
      else None,
  )

  # Start training
  logging.info('Starting training loop...')
  train_loop(
      af3_trainer=af3_trainer,
      train_dataset=train_dataset,
      val_dataset=val_dataset,
      config=training_config,
      output_dir=output_dir,
  )

  logging.info('Training finished!')


if __name__ == '__main__':
  flags.mark_flags_as_required(['train_data_dir', 'output_dir'])
  app.run(main)
