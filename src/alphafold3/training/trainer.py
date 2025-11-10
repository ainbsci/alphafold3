# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Training loop and optimizer for AlphaFold 3."""

from typing import Any, Dict, Optional, Tuple
import functools
import pathlib

from absl import logging
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from alphafold3.model import model as af3_model
from alphafold3.model import features
from alphafold3.model import params as model_params
from alphafold3.training import losses
from alphafold3.training import train_config


class AlphaFold3Trainer:
  """Trainer for AlphaFold 3 fine-tuning."""

  def __init__(
      self,
      model_config: af3_model.Model.Config,
      train_config: train_config.TrainingConfig,
      pretrained_params_dir: Optional[pathlib.Path] = None,
  ):
    """Initialize the trainer.

    Args:
      model_config: Model configuration
      train_config: Training configuration
      pretrained_params_dir: Directory containing pretrained model parameters
    """
    self.model_config = model_config
    self.train_config = train_config
    self.pretrained_params_dir = pretrained_params_dir

    # Initialize model
    self._init_model()

    # Initialize optimizer
    self._init_optimizer()

  def _init_model(self):
    """Initialize the Haiku-transformed model."""

    @hk.transform
    def forward_fn(batch: features.BatchDict):
      """Forward pass."""
      model = af3_model.Model(self.model_config)
      return model(batch)

    self.forward = forward_fn

  def _init_optimizer(self):
    """Initialize the optimizer with learning rate schedule."""
    opt_config = self.train_config.optimization

    # Learning rate schedule with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=opt_config.max_lr,
        warmup_steps=opt_config.warmup_steps,
        decay_steps=self.train_config.num_train_steps,
        end_value=opt_config.min_lr,
    )

    # AdamW optimizer with gradient clipping
    self.optimizer = optax.chain(
        optax.clip_by_global_norm(opt_config.gradient_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=opt_config.adam_beta1,
            b2=opt_config.adam_beta2,
            eps=opt_config.adam_eps,
            weight_decay=opt_config.weight_decay,
        ),
    )

  def load_pretrained_params(self) -> hk.Params:
    """Load pretrained model parameters.

    Returns:
      Haiku parameters dictionary
    """
    if self.pretrained_params_dir is None:
      raise ValueError('pretrained_params_dir must be set to load parameters')

    logging.info(f'Loading pretrained parameters from {self.pretrained_params_dir}')
    params = model_params.get_model_haiku_params(
        model_dir=self.pretrained_params_dir
    )

    # Apply parameter freezing if configured
    if self.train_config.fine_tuning.freeze_trunk:
      params = self._freeze_params(params, 'trunk')
    if self.train_config.fine_tuning.freeze_evoformer:
      params = self._freeze_params(params, 'evoformer')
    if self.train_config.fine_tuning.freeze_diffusion_transformer:
      params = self._freeze_params(params, 'diffusion_transformer')

    return params

  def _freeze_params(
      self, params: hk.Params, module_name: str
  ) -> hk.Params:
    """Mark certain parameters as frozen (for gradient masking).

    Note: In JAX, we don't actually freeze parameters, but we can mask
    gradients during the update step.

    Args:
      params: Parameter dictionary
      module_name: Name of module to freeze

    Returns:
      Parameters with frozen marker (implementation detail)
    """
    # Store frozen module names in metadata
    if '__frozen__' not in params:
      params['__frozen__'] = set()
    params['__frozen__'].add(module_name)
    return params

  @functools.partial(jax.jit, static_argnums=(0,))
  def train_step(
      self,
      params: hk.Params,
      opt_state: optax.OptState,
      batch: features.BatchDict,
      true_positions: jnp.ndarray,
      rng_key: jax.Array,
  ) -> Tuple[hk.Params, optax.OptState, Dict[str, jnp.ndarray]]:
    """Perform a single training step.

    Args:
      params: Current model parameters
      opt_state: Current optimizer state
      batch: Input batch
      true_positions: Ground truth atom positions
      rng_key: Random key for noise sampling

    Returns:
      updated_params: Updated parameters
      updated_opt_state: Updated optimizer state
      metrics: Dictionary of loss and metrics
    """

    def loss_fn(params, batch, true_positions, rng_key):
      """Compute loss for a batch."""
      # Forward pass
      output = self.forward.apply(params, rng_key, batch)

      # Compute losses
      total_loss, loss_dict = losses.compute_total_loss(
          model_output=output,
          batch=batch,
          true_positions=true_positions,
          loss_weights={
              'diffusion': self.train_config.loss_weights.diffusion,
              'plddt': self.train_config.loss_weights.plddt,
              'pae': self.train_config.loss_weights.pae,
              'distogram': self.train_config.loss_weights.distogram,
              'fape': self.train_config.loss_weights.fape,
          },
      )

      return total_loss, (loss_dict, output)

    # Compute gradients
    (total_loss, (loss_dict, output)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params, batch, true_positions, rng_key)

    # Apply gradient masking for frozen parameters
    if '__frozen__' in params:
      grads = self._mask_frozen_gradients(grads, params['__frozen__'])

    # Update parameters
    updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Compute gradient norm for monitoring
    grad_norm = optax.global_norm(grads)

    # Prepare metrics
    metrics = {
        'total_loss': total_loss,
        'grad_norm': grad_norm,
        **{f'loss/{k}': v for k, v in loss_dict.items()},
    }

    return new_params, new_opt_state, metrics

  def _mask_frozen_gradients(
      self, grads: hk.Params, frozen_modules: set
  ) -> hk.Params:
    """Mask gradients for frozen modules.

    Args:
      grads: Gradient tree
      frozen_modules: Set of frozen module names

    Returns:
      Gradients with frozen modules zeroed out
    """

    def zero_if_frozen(path, grad):
      """Zero out gradient if path matches frozen module."""
      for frozen_name in frozen_modules:
        if frozen_name in '/'.join(path):
          return jnp.zeros_like(grad)
      return grad

    # Use tree_map_with_path to check each parameter path
    return hk.data_structures.tree_map_with_path(zero_if_frozen, grads)

  @functools.partial(jax.jit, static_argnums=(0,))
  def eval_step(
      self,
      params: hk.Params,
      batch: features.BatchDict,
      true_positions: jnp.ndarray,
      rng_key: jax.Array,
  ) -> Dict[str, jnp.ndarray]:
    """Perform a single evaluation step.

    Args:
      params: Model parameters
      batch: Input batch
      true_positions: Ground truth positions
      rng_key: Random key

    Returns:
      metrics: Dictionary of evaluation metrics
    """
    # Forward pass
    output = self.forward.apply(params, rng_key, batch)

    # Compute losses (no gradients needed)
    total_loss, loss_dict = losses.compute_total_loss(
        model_output=output,
        batch=batch,
        true_positions=true_positions,
        loss_weights={
            'diffusion': self.train_config.loss_weights.diffusion,
            'plddt': self.train_config.loss_weights.plddt,
            'pae': self.train_config.loss_weights.pae,
            'distogram': self.train_config.loss_weights.distogram,
            'fape': self.train_config.loss_weights.fape,
        },
    )

    metrics = {
        'total_loss': total_loss,
        **{f'loss/{k}': v for k, v in loss_dict.items()},
    }

    return metrics

  def save_checkpoint(
      self,
      params: hk.Params,
      opt_state: optax.OptState,
      step: int,
      checkpoint_dir: pathlib.Path,
  ):
    """Save checkpoint.

    Args:
      params: Model parameters
      opt_state: Optimizer state
      step: Training step
      checkpoint_dir: Directory to save checkpoint
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f'checkpoint_{step}.pkl'

    # Save using Haiku's serialization (simplified)
    # In practice, you'd use a more robust checkpoint library
    import pickle

    checkpoint_data = {
        'params': params,
        'opt_state': opt_state,
        'step': step,
        'config': self.train_config,
    }

    with open(checkpoint_path, 'wb') as f:
      pickle.dump(checkpoint_data, f)

    logging.info(f'Saved checkpoint to {checkpoint_path}')

    # Clean up old checkpoints
    self._cleanup_old_checkpoints(
        checkpoint_dir, keep_n=self.train_config.checkpoint.keep_n_checkpoints
    )

  def _cleanup_old_checkpoints(
      self, checkpoint_dir: pathlib.Path, keep_n: int
  ):
    """Remove old checkpoints, keeping only the most recent N.

    Args:
      checkpoint_dir: Checkpoint directory
      keep_n: Number of checkpoints to keep
    """
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pkl'))
    if len(checkpoints) > keep_n:
      for old_checkpoint in checkpoints[:-keep_n]:
        old_checkpoint.unlink()
        logging.info(f'Removed old checkpoint: {old_checkpoint}')

  def load_checkpoint(
      self, checkpoint_path: pathlib.Path
  ) -> Tuple[hk.Params, optax.OptState, int]:
    """Load checkpoint.

    Args:
      checkpoint_path: Path to checkpoint file

    Returns:
      params: Model parameters
      opt_state: Optimizer state
      step: Training step
    """
    import pickle

    with open(checkpoint_path, 'rb') as f:
      checkpoint_data = pickle.load(f)

    logging.info(f'Loaded checkpoint from {checkpoint_path}')

    return (
        checkpoint_data['params'],
        checkpoint_data['opt_state'],
        checkpoint_data['step'],
    )
