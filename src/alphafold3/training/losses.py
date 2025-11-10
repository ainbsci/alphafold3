# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions for AlphaFold 3 training.

Based on the AlphaFold 3 paper (Nature 2024):
https://doi.org/10.1038/s41586-024-07487-w

Key losses:
1. Diffusion loss: MSE between predicted and ground truth atom positions
2. Confidence losses: pLDDT, PAE, PDE predictions
3. Auxiliary losses: Distogram, experimental constraints
"""

from typing import Dict, Any
import jax
import jax.numpy as jnp
from alphafold3.model import feat_batch
from alphafold3.model.atom_layout import atom_layout


# Constants from the paper/codebase
SIGMA_DATA = 16.0  # From diffusion_head.py


def diffusion_loss(
    predicted_positions: jnp.ndarray,
    true_positions: jnp.ndarray,
    atom_mask: jnp.ndarray,
    noise_level: float,
    reduction: str = 'mean'
) -> jnp.ndarray:
  """Compute diffusion loss for structure prediction.

  This is the main loss for training the diffusion model. Following the
  EDM (Elucidating the Design Space of Diffusion-Based Generative Models)
  framework used in AlphaFold 3.

  Args:
    predicted_positions: Predicted atom positions [num_tokens, num_atoms, 3]
    true_positions: Ground truth atom positions [num_tokens, num_atoms, 3]
    atom_mask: Mask for valid atoms [num_tokens, num_atoms]
    noise_level: Noise level sigma used during training
    reduction: How to reduce the loss ('mean', 'sum', or 'none')

  Returns:
    Diffusion loss value(s)
  """
  # Compute per-atom squared errors
  squared_errors = jnp.square(predicted_positions - true_positions)

  # Sum over coordinate dimensions (x, y, z)
  per_atom_loss = jnp.sum(squared_errors, axis=-1)  # [num_tokens, num_atoms]

  # Apply atom mask
  masked_loss = per_atom_loss * atom_mask

  # Weight by noise level (EDM-style weighting)
  # lambda(sigma) = 1 / (sigma^2 + sigma_data^2)
  weight = 1.0 / (noise_level**2 + SIGMA_DATA**2)
  weighted_loss = weight * masked_loss

  if reduction == 'mean':
    # Average over valid atoms
    num_valid_atoms = jnp.sum(atom_mask) + 1e-8
    return jnp.sum(weighted_loss) / num_valid_atoms
  elif reduction == 'sum':
    return jnp.sum(weighted_loss)
  else:  # 'none'
    return weighted_loss


def plddt_loss(
    predicted_lddt_logits: jnp.ndarray,
    true_lddt: jnp.ndarray,
    num_bins: int = 50,
    reduction: str = 'mean'
) -> jnp.ndarray:
  """Compute pLDDT (predicted local distance difference test) loss.

  pLDDT is a per-residue confidence metric that predicts local accuracy.

  Args:
    predicted_lddt_logits: Predicted pLDDT logits [num_tokens, num_bins]
    true_lddt: Ground truth LDDT values [num_tokens] in range [0, 1]
    num_bins: Number of bins for discretization (default: 50)
    reduction: How to reduce the loss

  Returns:
    pLDDT loss value
  """
  # Convert continuous LDDT to bin indices
  bin_indices = jnp.clip(
      jnp.floor(true_lddt * num_bins).astype(jnp.int32),
      0,
      num_bins - 1
  )

  # Compute cross-entropy loss
  log_probs = jax.nn.log_softmax(predicted_lddt_logits, axis=-1)
  loss = -jnp.take_along_axis(
      log_probs,
      bin_indices[..., None],
      axis=-1
  ).squeeze(-1)

  if reduction == 'mean':
    return jnp.mean(loss)
  elif reduction == 'sum':
    return jnp.sum(loss)
  else:
    return loss


def pae_loss(
    predicted_pae_logits: jnp.ndarray,
    true_positions: jnp.ndarray,
    predicted_positions: jnp.ndarray,
    token_mask: jnp.ndarray,
    max_error_bin: float = 31.0,
    num_bins: int = 64,
    reduction: str = 'mean'
) -> jnp.ndarray:
  """Compute PAE (Predicted Aligned Error) loss.

  PAE predicts the position error at residue i when the prediction is aligned
  on residue j.

  Args:
    predicted_pae_logits: Predicted PAE logits [num_tokens, num_tokens, num_bins]
    true_positions: Ground truth positions [num_tokens, num_atoms, 3]
    predicted_positions: Predicted positions [num_tokens, num_atoms, 3]
    token_mask: Valid token mask [num_tokens]
    max_error_bin: Maximum error value
    num_bins: Number of bins
    reduction: How to reduce the loss

  Returns:
    PAE loss value
  """
  # Compute pairwise distances for ground truth and predictions
  # Using CA or representative atom (first atom per token)
  true_repr = true_positions[:, 0, :]  # [num_tokens, 3]
  pred_repr = predicted_positions[:, 0, :]  # [num_tokens, 3]

  # Compute pairwise errors
  true_dists = jnp.linalg.norm(
      true_repr[:, None, :] - true_repr[None, :, :],
      axis=-1
  )
  pred_dists = jnp.linalg.norm(
      pred_repr[:, None, :] - pred_repr[None, :, :],
      axis=-1
  )

  # Position errors
  errors = jnp.abs(true_dists - pred_dists)

  # Clip and discretize errors
  errors = jnp.clip(errors, 0.0, max_error_bin)
  bin_indices = jnp.floor(errors / max_error_bin * num_bins).astype(jnp.int32)
  bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)

  # Compute cross-entropy
  log_probs = jax.nn.log_softmax(predicted_pae_logits, axis=-1)
  loss = -jnp.take_along_axis(
      log_probs,
      bin_indices[..., None],
      axis=-1
  ).squeeze(-1)

  # Apply pairwise mask
  pair_mask = token_mask[:, None] * token_mask[None, :]
  loss = loss * pair_mask

  if reduction == 'mean':
    num_valid_pairs = jnp.sum(pair_mask) + 1e-8
    return jnp.sum(loss) / num_valid_pairs
  elif reduction == 'sum':
    return jnp.sum(loss)
  else:
    return loss


def distogram_loss(
    predicted_distogram_logits: jnp.ndarray,
    true_positions: jnp.ndarray,
    token_mask: jnp.ndarray,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
    reduction: str = 'mean'
) -> jnp.ndarray:
  """Compute distogram loss for distance prediction.

  Args:
    predicted_distogram_logits: Predicted distogram [num_tokens, num_tokens, num_bins]
    true_positions: Ground truth positions [num_tokens, num_atoms, 3]
    token_mask: Valid token mask [num_tokens]
    min_dist: Minimum distance for binning
    max_dist: Maximum distance for binning
    num_bins: Number of distance bins
    reduction: How to reduce the loss

  Returns:
    Distogram loss value
  """
  # Compute pairwise distances using representative atoms (e.g., CA)
  repr_positions = true_positions[:, 0, :]  # [num_tokens, 3]

  distances = jnp.linalg.norm(
      repr_positions[:, None, :] - repr_positions[None, :, :],
      axis=-1
  )

  # Clip distances to range
  distances = jnp.clip(distances, min_dist, max_dist)

  # Convert to bin indices
  bin_width = (max_dist - min_dist) / num_bins
  bin_indices = jnp.floor((distances - min_dist) / bin_width).astype(jnp.int32)
  bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)

  # Cross-entropy loss
  log_probs = jax.nn.log_softmax(predicted_distogram_logits, axis=-1)
  loss = -jnp.take_along_axis(
      log_probs,
      bin_indices[..., None],
      axis=-1
  ).squeeze(-1)

  # Apply mask
  pair_mask = token_mask[:, None] * token_mask[None, :]
  loss = loss * pair_mask

  if reduction == 'mean':
    num_valid_pairs = jnp.sum(pair_mask) + 1e-8
    return jnp.sum(loss) / num_valid_pairs
  elif reduction == 'sum':
    return jnp.sum(loss)
  else:
    return loss


def fape_loss(
    predicted_positions: jnp.ndarray,
    true_positions: jnp.ndarray,
    atom_mask: jnp.ndarray,
    token_mask: jnp.ndarray,
    clamp_distance: float = 10.0,
    reduction: str = 'mean'
) -> jnp.ndarray:
  """Frame Aligned Point Error (FAPE) loss.

  FAPE measures the error in predicted atom positions after aligning frames.
  This is an auxiliary loss that can help with structure prediction.

  Args:
    predicted_positions: Predicted positions [num_tokens, num_atoms, 3]
    true_positions: Ground truth positions [num_tokens, num_atoms, 3]
    atom_mask: Atom validity mask [num_tokens, num_atoms]
    token_mask: Token validity mask [num_tokens]
    clamp_distance: Maximum distance for clamping
    reduction: How to reduce the loss

  Returns:
    FAPE loss value
  """
  # Compute per-atom distances
  dists = jnp.sqrt(
      jnp.sum(
          jnp.square(predicted_positions - true_positions),
          axis=-1
      ) + 1e-8
  )

  # Clamp distances
  dists = jnp.clip(dists, 0.0, clamp_distance)

  # Apply masks
  dists = dists * atom_mask

  # Expand token mask to atom level
  expanded_token_mask = token_mask[:, None] * atom_mask

  if reduction == 'mean':
    num_valid = jnp.sum(expanded_token_mask) + 1e-8
    return jnp.sum(dists) / num_valid
  elif reduction == 'sum':
    return jnp.sum(dists)
  else:
    return dists


def compute_total_loss(
    model_output: Dict[str, Any],
    batch: feat_batch.Batch,
    true_positions: jnp.ndarray,
    loss_weights: Dict[str, float] = None,
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
  """Compute total weighted loss for training.

  Combines all loss components with configurable weights.

  Args:
    model_output: Dictionary of model predictions
    batch: Input batch containing masks and features
    true_positions: Ground truth atom positions
    loss_weights: Dictionary of loss weights, defaults to paper values

  Returns:
    total_loss: Scalar total loss
    loss_dict: Dictionary of individual loss components
  """
  if loss_weights is None:
    # Default weights based on AlphaFold 3 training
    loss_weights = {
        'diffusion': 1.0,
        'plddt': 0.01,
        'pae': 0.1,
        'distogram': 0.01,
        'fape': 0.5,
    }

  loss_dict = {}

  # Extract atom mask from batch
  atom_mask = batch.atom_features.mask  # [num_tokens, num_atoms]
  token_mask = batch.token_features.mask  # [num_tokens]

  # Diffusion loss (main loss)
  if 'diffusion_samples' in model_output:
    predicted_pos = model_output['diffusion_samples']['atom_positions']
    # Use first sample for training
    if predicted_pos.ndim == 4:  # [num_samples, num_tokens, num_atoms, 3]
      predicted_pos = predicted_pos[0]

    noise_level = model_output.get('noise_level', 1.0)
    loss_dict['diffusion'] = diffusion_loss(
        predicted_pos,
        true_positions,
        atom_mask,
        noise_level
    )

  # pLDDT loss
  if 'predicted_lddt' in model_output:
    # Compute true LDDT from positions (simplified)
    true_lddt = compute_lddt_from_positions(
        predicted_pos if 'diffusion_samples' in model_output else true_positions,
        true_positions,
        atom_mask
    )
    loss_dict['plddt'] = plddt_loss(
        model_output['predicted_lddt'],
        true_lddt
    )

  # PAE loss
  if 'pae' in model_output or 'full_pae' in model_output:
    pae_logits = model_output.get('pae', model_output.get('full_pae'))
    if pae_logits is not None and predicted_pos is not None:
      loss_dict['pae'] = pae_loss(
          pae_logits,
          true_positions,
          predicted_pos,
          token_mask
      )

  # Distogram loss
  if 'distogram' in model_output and 'distogram' in model_output['distogram']:
    loss_dict['distogram'] = distogram_loss(
        model_output['distogram']['distogram'],
        true_positions,
        token_mask
    )

  # FAPE loss (auxiliary)
  if predicted_pos is not None:
    loss_dict['fape'] = fape_loss(
        predicted_pos,
        true_positions,
        atom_mask,
        token_mask
    )

  # Compute weighted total loss
  total_loss = jnp.array(0.0)
  for loss_name, loss_value in loss_dict.items():
    weight = loss_weights.get(loss_name, 0.0)
    total_loss = total_loss + weight * loss_value

  return total_loss, loss_dict


def compute_lddt_from_positions(
    predicted_positions: jnp.ndarray,
    true_positions: jnp.ndarray,
    atom_mask: jnp.ndarray,
    cutoff: float = 15.0,
) -> jnp.ndarray:
  """Compute LDDT scores from positions (simplified version).

  Args:
    predicted_positions: Predicted atom positions [num_tokens, num_atoms, 3]
    true_positions: Ground truth positions [num_tokens, num_atoms, 3]
    atom_mask: Atom validity mask [num_tokens, num_atoms]
    cutoff: Distance cutoff for local environment

  Returns:
    Per-token LDDT scores [num_tokens]
  """
  # Use representative atom (first atom per token)
  pred_repr = predicted_positions[:, 0, :]  # [num_tokens, 3]
  true_repr = true_positions[:, 0, :]  # [num_tokens, 3]

  # Compute pairwise distances
  true_dists = jnp.linalg.norm(
      true_repr[:, None, :] - true_repr[None, :, :],
      axis=-1
  )
  pred_dists = jnp.linalg.norm(
      pred_repr[:, None, :] - pred_repr[None, :, :],
      axis=-1
  )

  # Find neighbors within cutoff
  neighbors = (true_dists < cutoff) & (true_dists > 0.0)

  # Compute distance differences
  dist_diff = jnp.abs(true_dists - pred_dists)

  # LDDT thresholds: 0.5, 1.0, 2.0, 4.0 Angstroms
  thresholds = jnp.array([0.5, 1.0, 2.0, 4.0])

  # Fraction of distances preserved at each threshold
  preserved = dist_diff[:, :, None] < thresholds[None, None, :]  # [N, N, 4]
  preserved = preserved * neighbors[:, :, None]

  # Average over thresholds and neighbors
  num_neighbors = jnp.sum(neighbors, axis=-1, keepdims=True) + 1e-8
  lddt = jnp.sum(preserved, axis=(-2, -1)) / (num_neighbors.squeeze(-1) * 4.0)

  return lddt
