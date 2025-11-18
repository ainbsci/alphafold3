# AlphaFold 3 Training Framework: Algorithmic Deep Dive

**Technical Report**
**Date:** November 2025
**Version:** 1.0

---

## Executive Summary

This document provides a comprehensive technical explanation of the algorithms implemented in the AlphaFold 3 training framework. It covers the mathematical formulations, implementation details, and design decisions for training and fine-tuning AlphaFold 3 on custom datasets.

**Key Components:**
- Diffusion model training (EDM framework)
- Multi-objective loss functions
- Stochastic gradient optimization
- Data featurization pipeline

---

## Table of Contents

1. [Model Architecture Overview](#1-model-architecture-overview)
2. [Diffusion Model Framework](#2-diffusion-model-framework)
3. [Loss Functions](#3-loss-functions)
4. [Optimization Algorithm](#4-optimization-algorithm)
5. [Data Processing Pipeline](#5-data-processing-pipeline)
6. [Training Procedure](#6-training-procedure)
7. [Implementation Details](#7-implementation-details)
8. [Algorithmic Complexity](#8-algorithmic-complexity)

---

## 1. Model Architecture Overview

### 1.1 High-Level Architecture

AlphaFold 3 consists of three main components:

```
Input Features → Evoformer → Diffusion Head → 3D Structure
                    ↓
              Confidence Head → pLDDT, PAE, etc.
```

**Components:**

1. **Evoformer (Trunk)**
   - Processes MSA and pairwise features
   - Produces embeddings: `(single, pair)`
   - Configuration: 48 blocks, 384 single channels, 128 pair channels

2. **Diffusion Head**
   - Denoising diffusion model for atomic coordinates
   - Transformer-based architecture
   - Iterative refinement through reverse diffusion

3. **Confidence Head**
   - Predicts quality metrics (pLDDT, PAE, PDE)
   - PairFormer architecture (4 layers)
   - Outputs probabilistic confidence scores

### 1.2 Mathematical Notation

| Symbol | Description | Dimensions |
|--------|-------------|------------|
| `N` | Number of tokens (residues) | scalar |
| `A` | Number of atoms per token | scalar |
| `x ∈ ℝ^(N×A×3)` | Atomic coordinates | 3D positions |
| `s ∈ ℝ^(N×d_s)` | Single (token) embeddings | d_s = 384 |
| `z ∈ ℝ^(N×N×d_z)` | Pair embeddings | d_z = 128 |
| `σ` | Noise level | scalar |
| `t ∈ [0,1]` | Diffusion timestep | scalar |

---

## 2. Diffusion Model Framework

### 2.1 Elucidating the Design Space of Diffusion Models (EDM)

AlphaFold 3 uses the EDM framework for structure generation.

#### 2.1.1 Noise Schedule

**Definition:**
```
σ(t) = σ_data · (σ_max^(1/ρ) + t · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
```

**Parameters:**
- `σ_data = 16.0` Å (data noise scale, empirically measured)
- `σ_min = 0.0004` Å (minimum noise)
- `σ_max = 160.0` Å (maximum noise)
- `ρ = 7.0` (schedule shaping parameter)

**Purpose:** Controls the noise added at each diffusion step, following a polynomial schedule.

#### 2.1.2 Forward Process (Noising)

Given clean coordinates `x_0`, add Gaussian noise:

```
x_t = x_0 + σ(t) · ε,  where ε ~ 𝒩(0, I)
```

**Interpretation:** Gradually destroys structure by adding noise scaled by `σ(t)`.

#### 2.1.3 Reverse Process (Denoising)

The model learns to predict the denoised coordinates:

```
D_θ(x_t, σ(t), s, z) → x̂_0
```

Where:
- `D_θ` is the denoising network (Diffusion Transformer)
- `x_t` is noisy coordinates
- `σ(t)` is noise level embedding
- `s, z` are conditioning embeddings from Evoformer

#### 2.1.4 Sampling Algorithm

During inference, iteratively denoise from pure noise:

```python
Algorithm: Stochastic Sampler (EDM)

Input: Embeddings (s, z), number of steps T
Output: Predicted structure x_0

1. Initialize: x_T ~ 𝒩(0, σ_max² I)
2. For t = T-1 down to 0:
     a. σ_t = σ(t/T)
     b. x̂_0 = D_θ(x_t, σ_t, s, z)
     c. x_t-1 = x̂_0 + σ_t-1 · ε,  ε ~ 𝒩(0, I)
3. Return x_0
```

**Key Insight:** Each step predicts the clean structure `x̂_0`, then adds noise for the next iteration.

### 2.2 Training Objective

The diffusion loss minimizes the prediction error weighted by noise level:

```
𝓛_diffusion = 𝔼_{x_0, σ, ε} [λ(σ) · ||D_θ(x_0 + σε, σ, s, z) - x_0||²]
```

**Weighting function:**
```
λ(σ) = 1 / (σ² + σ_data²)
```

**Rationale:**
- Higher weight at low noise (near data distribution)
- Lower weight at high noise (structure is mostly destroyed)
- Balances learning across noise levels

---

## 3. Loss Functions

### 3.1 Diffusion Loss (Primary)

**Mathematical Formulation:**

```
𝓛_diff(θ) = (1 / (σ² + σ_data²)) · Σ_{i,j} m_{ij} · ||x̂_{ij} - x_{ij}||²

where:
  x̂_{ij} = predicted position of atom j in token i
  x_{ij} = ground truth position
  m_{ij} = atom mask (1 if atom exists, 0 otherwise)
  σ ~ p(σ) sampled from noise distribution
```

**Implementation Details:**

```python
def diffusion_loss(predicted, true, mask, sigma):
    # Per-atom squared error
    squared_error = (predicted - true)²
    # Shape: [N, A, 3]

    # Sum over coordinates (x, y, z)
    per_atom_loss = sum(squared_error, axis=-1)
    # Shape: [N, A]

    # Apply mask
    masked_loss = per_atom_loss * mask

    # Noise-dependent weighting
    lambda_sigma = 1.0 / (sigma² + sigma_data²)
    weighted_loss = lambda_sigma * masked_loss

    # Average over valid atoms
    return sum(weighted_loss) / sum(mask)
```

**Properties:**
- Differentiable w.r.t. model parameters θ
- Scale-invariant due to normalization
- Robust to missing atoms via masking

### 3.2 pLDDT Loss (Confidence)

**Predicted Local Distance Difference Test** - per-residue quality metric.

**Mathematical Formulation:**

Given ground truth LDDT score `y_i ∈ [0, 1]` for token `i`:

```
LDDT_i = (1/4|𝒩_i|) Σ_{j∈𝒩_i} Σ_{k∈{0.5,1,2,4}} 𝟙(|d_ij - d̂_ij| < k)

where:
  𝒩_i = {j : ||x_i - x_j|| < 15Å} (neighbors within 15Å)
  d_ij = ||x_i - x_j|| (true distance)
  d̂_ij = ||x̂_i - x̂_j|| (predicted distance)
  𝟙(·) = indicator function
```

**Discretization:**

Convert continuous LDDT to discrete bins:

```
bin(LDDT) = ⌊LDDT · K⌋,  K = 50 bins
```

**Loss (Cross-Entropy):**

```
𝓛_pLDDT = -Σ_i log(P_θ(bin(LDDT_i) | logits_i))

where:
  logits_i ∈ ℝ^K are predicted logits for token i
  P_θ(·) = softmax(logits_i)
```

**Implementation:**

```python
def plddt_loss(logits, true_lddt, num_bins=50):
    # Convert LDDT [0,1] to bin indices [0, num_bins-1]
    bin_indices = floor(true_lddt * num_bins)
    bin_indices = clip(bin_indices, 0, num_bins - 1)

    # Cross-entropy
    log_probs = log_softmax(logits, axis=-1)
    loss = -log_probs[bin_indices]

    return mean(loss)
```

**Interpretation:**
- Measures how well the model predicts its own accuracy
- Higher pLDDT → higher predicted accuracy
- Used for ranking predictions

### 3.3 PAE Loss (Predicted Aligned Error)

**Definition:** Expected position error at residue `i` when structure is aligned on residue `j`.

**Mathematical Formulation:**

```
PAE_{ij} = 𝔼[||x_i - T_j(x̂_i)||]

where:
  T_j(·) is optimal alignment based on neighborhood of residue j
```

**Approximation for Training:**

```
PAE_{ij} ≈ | ||x_i - x_j|| - ||x̂_i - x̂_j|| |

(difference in pairwise distances)
```

**Discretization:**

```
bin(PAE_{ij}) = ⌊(PAE_{ij} / max_error) · K⌋,  K = 64 bins, max_error = 31Å
```

**Loss:**

```
𝓛_PAE = -Σ_{i,j} m_i · m_j · log(P_θ(bin(PAE_{ij}) | logits_{ij}))

where:
  logits_{ij} ∈ ℝ^K are predicted logits for pair (i,j)
  m_i, m_j are token masks
```

**Implementation:**

```python
def pae_loss(logits, predicted_pos, true_pos, mask, num_bins=64, max_error=31.0):
    # Compute representative positions (e.g., Cα)
    pred_repr = predicted_pos[:, 0, :]  # [N, 3]
    true_repr = true_pos[:, 0, :]

    # Pairwise distances
    true_dists = norm(true_repr[:, None] - true_repr[None, :])  # [N, N]
    pred_dists = norm(pred_repr[:, None] - pred_repr[None, :])

    # Position errors
    errors = abs(true_dists - pred_dists)
    errors = clip(errors, 0, max_error)

    # Discretize
    bin_indices = floor((errors / max_error) * num_bins)
    bin_indices = clip(bin_indices, 0, num_bins - 1)

    # Cross-entropy with pairwise mask
    log_probs = log_softmax(logits, axis=-1)
    loss = -log_probs[bin_indices]
    pair_mask = mask[:, None] * mask[None, :]

    return sum(loss * pair_mask) / sum(pair_mask)
```

**Use Cases:**
- Evaluating prediction quality
- Understanding which regions are well-predicted
- ipTM (interface pTM) for complex interfaces

### 3.4 Distogram Loss

**Purpose:** Predict distribution of pairwise distances.

**Mathematical Formulation:**

```
𝓛_distogram = -Σ_{i,j} m_i · m_j · log(P_θ(bin(d_{ij}) | logits_{ij}))

where:
  d_{ij} = ||x_i - x_j|| (Cα-Cα distance)
  bins cover [2Å, 22Å] with 64 bins
```

**Distance Bins:**

```
bin_edges = linspace(2.0, 22.0, num_bins + 1)
bin(d) = argmin_k |d - bin_edges[k]|
```

**Implementation:**

```python
def distogram_loss(logits, true_pos, mask,
                   min_dist=2.0, max_dist=22.0, num_bins=64):
    # Representative atom distances
    repr_pos = true_pos[:, 0, :]
    distances = norm(repr_pos[:, None] - repr_pos[None, :])

    # Clip to range
    distances = clip(distances, min_dist, max_dist)

    # Bin assignment
    bin_width = (max_dist - min_dist) / num_bins
    bin_indices = floor((distances - min_dist) / bin_width)
    bin_indices = clip(bin_indices, 0, num_bins - 1)

    # Cross-entropy
    log_probs = log_softmax(logits, axis=-1)
    loss = -log_probs[bin_indices]
    pair_mask = mask[:, None] * mask[None, :]

    return sum(loss * pair_mask) / sum(pair_mask)
```

**Rationale:**
- Auxiliary supervision signal
- Helps learn spatial relationships
- Complements distance-based confidence metrics

### 3.5 FAPE Loss (Frame Aligned Point Error)

**Frame Aligned Point Error** - measures coordinate error after local alignment.

**Mathematical Formulation:**

For each token `i`, define a local frame `F_i`:

```
F_i = local coordinate system centered at token i

FAPE = Σ_i Σ_j∈𝒩_i Σ_a clamp(||F_i(x̂_{ja}) - F_i(x_{ja})||, 0, d_clamp)

where:
  𝒩_i = neighborhood of token i
  a indexes atoms
  F_i(·) transforms coordinates to frame i
  d_clamp = 10Å (clamping distance)
```

**Simplified Implementation (No Frame Alignment):**

For training, we use a simplified version without explicit frames:

```python
def fape_loss(predicted_pos, true_pos, atom_mask, token_mask, clamp=10.0):
    # Per-atom distances
    dists = sqrt(sum((predicted_pos - true_pos)², axis=-1) + 1e-8)

    # Clamp large errors
    dists = clip(dists, 0, clamp)

    # Apply masks
    dists = dists * atom_mask
    expanded_mask = token_mask[:, None] * atom_mask

    return sum(dists) / sum(expanded_mask)
```

**Purpose:**
- Focuses on local structure accuracy
- Robust to global alignment errors
- Useful for multi-domain proteins

### 3.6 Combined Loss

**Total Training Objective:**

```
𝓛_total = w_diff · 𝓛_diffusion
        + w_plddt · 𝓛_pLDDT
        + w_pae · 𝓛_PAE
        + w_dist · 𝓛_distogram
        + w_fape · 𝓛_FAPE
```

**Default Weights:**

```python
weights = {
    'diffusion': 1.0,    # Primary structure loss
    'plddt': 0.01,       # Confidence (low weight)
    'pae': 0.1,          # Alignment error
    'distogram': 0.01,   # Distance auxiliary
    'fape': 0.5,         # Local structure
}
```

**Rationale:**
- Diffusion loss is primary (weight = 1.0)
- Confidence losses are auxiliary (lower weights)
- Balances structural accuracy with quality prediction

---

## 4. Optimization Algorithm

### 4.1 AdamW Optimizer

**Algorithm:**

```
AdamW Optimization

Hyperparameters:
  α = learning rate (time-varying)
  β₁ = 0.9 (exponential decay rate for first moment)
  β₂ = 0.999 (exponential decay rate for second moment)
  ε = 1e-8 (numerical stability)
  λ = 1e-5 (weight decay)

Initialize:
  m₀ = 0 (first moment)
  v₀ = 0 (second moment)

For t = 1, 2, ..., T:
  1. Compute gradients:
     g_t = ∇_θ 𝓛_total(θ_t)

  2. Gradient clipping:
     if ||g_t|| > clip_norm:
       g_t = g_t · (clip_norm / ||g_t||)

  3. Update biased first moment:
     m_t = β₁ · m_{t-1} + (1 - β₁) · g_t

  4. Update biased second moment:
     v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²

  5. Bias correction:
     m̂_t = m_t / (1 - β₁^t)
     v̂_t = v_t / (1 - β₂^t)

  6. Parameter update:
     θ_{t+1} = θ_t - α_t · (m̂_t / (√v̂_t + ε) + λ · θ_t)
                              \_________________/   \______/
                                    Adam step       Weight decay
```

**Key Differences from Adam:**
- Weight decay applied directly to parameters (not to gradient)
- Decouples weight decay from gradient-based update
- Better generalization in practice

### 4.2 Learning Rate Schedule

**Warmup + Cosine Decay:**

```
α(t) = {
  α_max · (t / t_warmup)                           if t ≤ t_warmup
  α_min + (α_max - α_min) · (1 + cos(π·t'/T')) / 2 if t > t_warmup
}

where:
  t_warmup = 1000 (warmup steps)
  t' = t - t_warmup
  T' = T_total - t_warmup
  α_max = 1e-4 (peak learning rate)
  α_min = 1e-6 (minimum learning rate)
```

**Visualization:**

```
α(t) |     ___________________
     |    /                   \
     |   /                     \___
     |  /                          \___
     | /                              \_____
     |/                                     \_____
     |_______________________________________________ t
       warmup         plateau          decay
      (1000 steps)
```

**Rationale:**
- **Warmup:** Prevents large gradient updates at initialization
- **Cosine decay:** Smooth transition to smaller learning rates
- **Minimum LR:** Continues fine-tuning at end of training

### 4.3 Gradient Clipping

**Global Norm Clipping:**

```
clip_gradient(g, max_norm=1.0):
  global_norm = sqrt(Σ ||g_i||²)  (across all parameters)

  if global_norm > max_norm:
    g = g · (max_norm / global_norm)

  return g
```

**Purpose:**
- Prevents exploding gradients
- Stabilizes training of deep networks
- Critical for diffusion model stability

---

## 5. Data Processing Pipeline

### 5.1 Input Featurization

**From Structure to Features:**

```
Structure (.cif) → Parsing → Features → Model Input

Steps:
1. Parse mmCIF file
2. Extract chains and sequences
3. Compute MSA (or load pre-computed)
4. Search templates (or load pre-computed)
5. Featurize into tensor representation
6. Apply bucketing for compilation efficiency
```

### 5.2 Feature Tensors

**Key Features:**

| Feature | Shape | Description |
|---------|-------|-------------|
| `residue_index` | `[N]` | Sequential residue numbering |
| `aatype` | `[N]` | Amino acid type (0-21) |
| `atom_positions` | `[N, A, 3]` | Ground truth coordinates |
| `atom_mask` | `[N, A]` | Valid atom indicators |
| `msa` | `[M, N]` | Multiple sequence alignment |
| `msa_mask` | `[M, N]` | Valid MSA positions |
| `template_features` | `[T, N, N, F]` | Template pairwise features |
| `asym_id` | `[N]` | Chain identifiers |

Where:
- `N` = number of tokens
- `A` = max atoms per token (~37 for standard residues)
- `M` = number of MSA sequences
- `T` = number of templates
- `F` = number of template features

### 5.3 Bucketing Strategy

**Purpose:** Avoid excessive JAX recompilation for different sequence lengths.

**Algorithm:**

```
bucketing(num_tokens, buckets=[256, 512, 768, 1024, ...]):
  # Find smallest bucket that fits
  for bucket_size in buckets:
    if num_tokens <= bucket_size:
      return bucket_size

  # If larger than all buckets, create exact-size bucket
  return num_tokens
```

**Padding:**

```
pad_to_bucket(features, target_size):
  current_size = features['num_tokens']
  pad_size = target_size - current_size

  # Pad each feature tensor
  for key, tensor in features.items():
    if key has token dimension:
      tensor = concat([tensor, zeros([pad_size, ...])])

  # Update masks to mark padded positions as invalid
  features['mask'] = [1]*current_size + [0]*pad_size

  return features
```

**Trade-off:**
- Fewer buckets → less compilation, more padding waste
- More buckets → more compilation, less padding waste
- Default: 13 buckets covering [256, 5120] tokens

---

## 6. Training Procedure

### 6.1 Training Loop (High-Level)

```
Algorithm: AlphaFold3 Fine-Tuning

Input:
  - Pretrained parameters θ₀
  - Training dataset 𝒟_train = {(x_i, features_i)}
  - Hyperparameters (learning rate schedule, loss weights, etc.)

Output: Fine-tuned parameters θ*

1. Initialize:
   θ = θ₀ (load pretrained weights)
   opt_state = optimizer.init(θ)
   step = 0

2. While step < max_steps:

   a. Sample batch:
      (x_true, features) = sample(𝒟_train)

   b. Sample noise level:
      σ ~ Uniform(σ_min, σ_max)  or from schedule

   c. Featurize:
      batch = featurize(features, buckets)

   d. Forward pass:
      embeddings = Evoformer(batch)
      x_noisy = x_true + σ · ε,  ε ~ 𝒩(0,I)
      x_pred = DiffusionHead(x_noisy, σ, embeddings)
      confidence = ConfidenceHead(x_pred, embeddings)
      distogram = DistogramHead(embeddings)

   e. Compute losses:
      𝓛_diff = diffusion_loss(x_pred, x_true, σ)
      𝓛_plddt = plddt_loss(confidence['plddt'], compute_lddt(x_pred, x_true))
      𝓛_pae = pae_loss(confidence['pae'], x_pred, x_true)
      𝓛_dist = distogram_loss(distogram, x_true)
      𝓛_total = combine_losses([𝓛_diff, 𝓛_plddt, 𝓛_pae, 𝓛_dist])

   f. Backward pass:
      g = ∇_θ 𝓛_total

   g. Gradient clipping:
      g = clip_gradient(g, max_norm=1.0)

   h. Apply freezing mask:
      if freeze_trunk:
        g[trunk_params] = 0

   i. Optimization step:
      updates, opt_state = optimizer.update(g, opt_state, θ)
      θ = apply_updates(θ, updates)

   j. Logging:
      if step % log_interval == 0:
        log(step, 𝓛_total, grad_norm)

   k. Checkpointing:
      if step % checkpoint_interval == 0:
        save_checkpoint(θ, opt_state, step)

   l. Validation:
      if step % eval_interval == 0:
        eval_loss = evaluate(θ, 𝒟_val)
        log(step, eval_loss)

   step += 1

3. Return θ
```

### 6.2 Single Training Step (Detailed)

**JAX Implementation:**

```python
@jax.jit
def train_step(params, opt_state, batch, true_positions, rng_key):
    """
    Single training step with automatic differentiation.

    Args:
      params: Model parameters (Haiku params dict)
      opt_state: Optimizer state
      batch: Featurized input batch
      true_positions: Ground truth atom coordinates [N, A, 3]
      rng_key: JAX random key

    Returns:
      updated_params, updated_opt_state, metrics
    """

    def loss_fn(params):
        # Forward pass
        rng_1, rng_2 = jax.random.split(rng_key)

        # Sample noise level
        sigma = sample_noise_level(rng_1)

        # Add noise to true positions
        noise = jax.random.normal(rng_2, true_positions.shape)
        noisy_positions = true_positions + sigma * noise

        # Model forward
        output = model.apply(params, batch, noisy_positions, sigma)

        # Compute losses
        total_loss, loss_dict = compute_total_loss(
            output, batch, true_positions, sigma
        )

        return total_loss, loss_dict

    # Compute gradients
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Gradient clipping
    grads = clip_by_global_norm(grads, max_norm=1.0)

    # Optimizer update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Metrics
    metrics = {
        'loss': loss,
        'grad_norm': global_norm(grads),
        **aux
    }

    return new_params, new_opt_state, metrics
```

**Key Aspects:**

1. **JIT Compilation:** Entire step is JIT-compiled for performance
2. **Automatic Differentiation:** JAX computes gradients automatically
3. **Functional:** No side effects, pure functions
4. **Efficient:** On-device computation, minimal host transfers

### 6.3 Noise Level Sampling

**Training Strategy:**

During training, we sample noise levels from the schedule:

```python
def sample_noise_level(rng_key, strategy='log_uniform'):
    """
    Sample noise level for training step.

    Strategies:
      - 'uniform': σ ~ Uniform(σ_min, σ_max)
      - 'log_uniform': log(σ) ~ Uniform(log(σ_min), log(σ_max))
      - 'schedule': σ(t), t ~ Uniform(0, 1)
    """

    if strategy == 'log_uniform':
        # Sample in log space for better coverage
        log_sigma_min = log(0.0004)
        log_sigma_max = log(160.0)
        log_sigma = jax.random.uniform(rng_key, [],
                                       minval=log_sigma_min,
                                       maxval=log_sigma_max)
        return exp(log_sigma)

    elif strategy == 'schedule':
        # Sample timestep, then apply noise schedule
        t = jax.random.uniform(rng_key, [])
        return noise_schedule(t)

    else:  # uniform
        return jax.random.uniform(rng_key, [],
                                  minval=0.0004,
                                  maxval=160.0)
```

**Rationale:**
- Log-uniform sampling ensures coverage of all noise scales
- Important for EDM: need to train across entire noise range
- Different noise levels teach different aspects of structure

---

## 7. Implementation Details

### 7.1 JAX and Haiku Integration

**Haiku Transform Pattern:**

```python
def create_model():
    @hk.transform
    def forward(batch):
        model = AlphaFold3Model(config)
        return model(batch)

    return forward

# Usage
model = create_model()
params = model.init(rng_key, sample_batch)
output = model.apply(params, rng_key, batch)
```

**Key Concepts:**
- **Transform:** Converts stateful module to pure functions
- **init:** Initializes parameters from random key + sample input
- **apply:** Runs forward pass with given parameters
- **Functional:** No hidden state, all explicit

### 7.2 Gradient Masking for Freezing

**Selective Freezing:**

```python
def mask_frozen_gradients(grads, frozen_modules):
    """
    Zero out gradients for frozen modules.

    Args:
      grads: Gradient tree (nested dict)
      frozen_modules: Set of module names to freeze

    Returns:
      Masked gradient tree
    """

    def zero_if_frozen(path, grad):
        # Check if any frozen module name is in path
        path_str = '/'.join(path)
        for frozen_name in frozen_modules:
            if frozen_name in path_str:
                return jnp.zeros_like(grad)
        return grad

    return hk.data_structures.tree_map_with_path(
        zero_if_frozen, grads
    )
```

**Example:**

```python
# Freeze evoformer trunk
frozen = {'evoformer'}
grads = compute_gradients(...)
grads = mask_frozen_gradients(grads, frozen)
# Now evoformer gradients are all zeros
```

### 7.3 Checkpointing

**Checkpoint Format:**

```python
checkpoint = {
    'params': params,           # Model parameters (nested dict)
    'opt_state': opt_state,     # Optimizer state (momentum, etc.)
    'step': step,               # Current training step
    'config': config,           # Training configuration
    'rng_state': rng_key,       # Random state (for reproducibility)
}
```

**Save/Load:**

```python
def save_checkpoint(checkpoint, path):
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
```

**Checkpoint Management:**

```python
def cleanup_old_checkpoints(checkpoint_dir, keep_n=5):
    """Keep only the N most recent checkpoints."""
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pkl'))
    if len(checkpoints) > keep_n:
        for old_ckpt in checkpoints[:-keep_n]:
            old_ckpt.unlink()
```

### 7.4 Memory Optimization

**Techniques Used:**

1. **Gradient Checkpointing:**
   ```python
   # Not implemented yet, but would use:
   # hk.remat for recomputing activations in backward pass
   ```

2. **Mixed Precision (bfloat16):**
   ```python
   if config.use_bfloat16:
       embeddings = embeddings.astype(jnp.bfloat16)
       # Reduces memory by 2x for large tensors
   ```

3. **Batch Size 1:**
   - Limitation: Variable sequence lengths
   - Benefit: Lower memory usage
   - Future: Dynamic batching with padding

4. **Efficient Attention:**
   - FlashAttention (Triton kernel)
   - Reduces memory from O(N²) to O(N)
   - Critical for long sequences

---

## 8. Algorithmic Complexity

### 8.1 Time Complexity

**Per Training Step:**

| Component | Complexity | Notes |
|-----------|------------|-------|
| Evoformer | O(N²d + N²Md) | N=tokens, M=MSA depth, d=channels |
| Diffusion Transformer | O(N²d²) | Self-attention dominant |
| Confidence Head | O(N²d) | PairFormer (4 layers) |
| Loss Computation | O(N²) | Pairwise losses (PAE, distogram) |
| Gradient Computation | ~2× forward | Automatic differentiation |
| **Total** | **O(N²d²)** | Dominated by transformer |

**For N=512, d=384:**
- Forward: ~1.5 seconds on A100
- Backward: ~3 seconds
- Total: ~4.5 seconds per step

### 8.2 Space Complexity

**Memory Usage:**

| Component | Memory | Formula |
|-----------|--------|---------|
| Input Features | O(N²d) | Pair embeddings |
| Activations | O(N²d) | Intermediate tensors |
| Gradients | O(P) | P = total parameters (~millions) |
| Optimizer State | O(2P) | Adam: momentum + velocity |
| **Total** | **O(N²d + P)** | Dominated by activations for large N |

**Example (N=512, P=100M params):**
- Activations: ~40 GB (main bottleneck)
- Parameters: ~400 MB (fp32)
- Optimizer: ~800 MB (2× params)
- Total: ~41 GB (fits in A100 80GB)

### 8.3 Scaling Analysis

**Token Scaling:**

For sequences with N tokens:

```
Time: T(N) ∝ N²
Memory: M(N) ∝ N²
```

**Implication:** Quadratic scaling limits maximum sequence length.

**Mitigation:**
- Bucketing (compilation efficiency)
- Gradient checkpointing (future)
- Sparse attention (future)
- Model parallelism (future)

### 8.4 Training Duration Estimates

**Dataset Size vs. Training Time:**

| Dataset | Steps | A100 Time | H100 Time |
|---------|-------|-----------|-----------|
| 100 structures | 1,000 | ~1.5 hours | ~0.8 hours |
| 1,000 structures | 10,000 | ~12 hours | ~6 hours |
| 10,000 structures | 100,000 | ~5 days | ~2.5 days |

**Assumptions:**
- 4 seconds/step on A100, 2 seconds/step on H100
- Single GPU
- No multi-GPU parallelism

---

## 9. Theoretical Foundation

### 9.1 Diffusion Models Background

**Score-Based Generative Models:**

AlphaFold 3's diffusion approach is based on score-based models:

```
Score function: s_θ(x, t) = ∇_x log p_t(x)

Denoising score matching:
  𝓛 = 𝔼_{t, x_0, ε} [||s_θ(x_t, t) - ∇_x log p(x_t | x_0)||²]
```

**Connection to EDM:**

```
Denoiser: D_θ(x_t, σ)
Score: s_θ(x_t, σ) = (D_θ(x_t, σ) - x_t) / σ²

Optimal denoiser: D_θ(x_t, σ) = 𝔼[x_0 | x_t]
```

**Interpretation:**
- Model learns to predict clean data from noisy input
- Equivalent to learning the gradient of data distribution
- Sampling = reversing the noising process

### 9.2 Why EDM for Protein Structure?

**Advantages:**

1. **Continuous Space:** Atomic coordinates are continuous
2. **Multimodal:** Can represent multiple conformations
3. **Uncertainty:** Built-in uncertainty quantification
4. **Iterative Refinement:** Gradual denoising = gradual refinement

**Comparison to Alternatives:**

| Method | Pros | Cons |
|--------|------|------|
| **Diffusion (EDM)** | Flexible, high quality | Slower sampling |
| **Direct regression** | Fast | No uncertainty, mode collapse |
| **VAE** | Fast sampling | Blurry outputs |
| **GAN** | High quality | Training instability |

### 9.3 Loss Function Justification

**Multi-Task Learning:**

```
𝓛_total = Σ_i w_i · 𝓛_i

Rationale:
  - Different losses capture different aspects
  - Structure (diffusion) + Quality (confidence) + Geometry (distogram)
  - Weighted combination balances objectives
```

**Empirical Weights:**

Chosen based on:
1. Relative scale of losses
2. Importance to final metric (e.g., TM-score)
3. Stability during training

---

## 10. Advanced Topics

### 10.1 Data Augmentation

**Random Rigid Transformations:**

During training, apply random rotations + translations:

```python
def random_augmentation(positions, rng_key):
    # Random rotation (Gram-Schmidt orthogonalization)
    v0, v1 = random.normal(rng_key, (2, 3))
    e0 = v0 / norm(v0)
    e1 = v1 - e0 * dot(v1, e0)
    e1 = e1 / norm(e1)
    e2 = cross(e0, e1)
    R = stack([e0, e1, e2])  # Rotation matrix

    # Random translation
    t = random.normal(rng_key, (3,))

    # Apply transformation
    # Center structure, rotate, translate
    center = mean(positions, axis=(0,1))
    positions = (positions - center) @ R.T + t

    return positions
```

**Purpose:**
- SE(3) invariance: Model should be rotation/translation invariant
- Prevents overfitting to specific orientations
- Increases effective dataset size

### 10.2 Curriculum Learning (Future)

**Progressive Noise Scheduling:**

```
Idea: Start with easier (lower noise) examples, gradually increase difficulty

Curriculum:
  Steps 0-1000: σ ∈ [0.01, 10]  (low noise)
  Steps 1000-5000: σ ∈ [0.1, 50]  (medium noise)
  Steps 5000+: σ ∈ [0.0004, 160]  (full range)
```

**Benefits:**
- Faster convergence
- Better stability
- Improved final performance

### 10.3 Multi-GPU Training (Future)

**Data Parallelism with pmap:**

```python
# Replicate model across devices
@jax.pmap
def train_step_parallel(params, batch):
    # Each device gets a slice of the batch
    # Gradients are averaged across devices
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    return loss, grads

# Usage
devices = jax.local_devices()  # [GPU:0, GPU:1, ...]
params = jax.device_put_replicated(params, devices)

for batch in data_loader:
    # Split batch across devices
    batch_per_device = split(batch, num_devices)
    loss, grads = train_step_parallel(params, batch_per_device)
    # Gradients are automatically synced across devices
```

**Speedup:** ~linear with number of GPUs (if communication overhead is low)

---

## 11. Validation and Metrics

### 11.1 Evaluation Metrics

**Structure Quality:**

1. **RMSD (Root Mean Square Deviation):**
   ```
   RMSD = sqrt((1/N) Σ_i ||x̂_i - x_i||²)
   ```

2. **TM-score (Template Modeling score):**
   ```
   TM = (1/N) Σ_i (1 / (1 + (d_i/d_0)²))
   where d_0 = 1.24 · ∛(N - 15) - 1.8
   ```

3. **lDDT (local Distance Difference Test):**
   ```
   Fraction of preserved local distances
   Range: [0, 1], higher is better
   ```

**Confidence Quality:**

1. **pLDDT-lDDT Correlation:**
   ```
   Pearson correlation between predicted and true LDDT
   ```

2. **PAE Calibration:**
   ```
   Expected calibration error between predicted and actual errors
   ```

### 11.2 Validation Protocol

```
Algorithm: Model Validation

Input: Trained model θ, validation set 𝒟_val

1. For each example in 𝒟_val:
   a. Run inference: x̂ = model(features; θ)
   b. Compute structure metrics:
      - RMSD(x̂, x_true)
      - TM-score(x̂, x_true)
      - lDDT(x̂, x_true)
   c. Compute confidence metrics:
      - Correlation(pLDDT, lDDT)
      - Calibration(PAE)

2. Aggregate metrics:
   - Mean, median, std dev
   - Per-category breakdown (if applicable)

3. Compare to baseline:
   - Pretrained model performance
   - AlphaFold 2 (if applicable)

4. Return validation report
```

---

## 12. Limitations and Future Work

### 12.1 Current Limitations

1. **Batch Size:** Only 1 due to variable sequence lengths
2. **Multi-GPU:** Not implemented (requires pmap)
3. **From-Scratch Training:** Not validated, designed for fine-tuning
4. **Memory:** Requires A100 80GB for sequences > 512 tokens
5. **Speed:** ~4 seconds/step (could be optimized)

### 12.2 Future Improvements

**Algorithmic:**
- [ ] Implement curriculum learning
- [ ] Add more data augmentation strategies
- [ ] Experiment with different noise schedules
- [ ] Try alternative optimizers (Lion, AdaFactor)

**Engineering:**
- [ ] Multi-GPU data parallelism
- [ ] Gradient checkpointing for memory
- [ ] Dynamic batching with padding
- [ ] Mixed precision (bfloat16) everywhere
- [ ] Faster data loading (pre-featurization)

**Experimental:**
- [ ] Validate on large-scale benchmarks
- [ ] Ablation studies on loss weights
- [ ] Compare to other fine-tuning approaches
- [ ] Domain adaptation experiments

---

## 13. References

### Primary Literature

1. **AlphaFold 3 Paper:**
   - Abramson et al., "Accurate structure prediction of biomolecular interactions with AlphaFold 3", Nature (2024)
   - https://doi.org/10.1038/s41586-024-07487-w

2. **EDM (Diffusion Models):**
   - Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS (2022)
   - https://arxiv.org/abs/2206.00364

3. **Score-Based Models:**
   - Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution", NeurIPS (2019)
   - https://arxiv.org/abs/1907.05600

### Technical Documentation

4. **JAX Documentation:**
   - https://jax.readthedocs.io/

5. **Haiku Documentation:**
   - https://dm-haiku.readthedocs.io/

6. **Optax (Optimization):**
   - https://optax.readthedocs.io/

---

## Appendix A: Mathematical Notation Summary

| Symbol | Description | Type |
|--------|-------------|------|
| `N` | Number of tokens (residues) | Scalar |
| `A` | Atoms per token | Scalar |
| `x ∈ ℝ^(N×A×3)` | Atomic coordinates | Tensor |
| `σ` | Noise level | Scalar |
| `t ∈ [0,1]` | Diffusion time | Scalar |
| `θ` | Model parameters | Parameters |
| `s ∈ ℝ^(N×d_s)` | Single embeddings | Tensor |
| `z ∈ ℝ^(N×N×d_z)` | Pair embeddings | Tensor |
| `D_θ` | Denoising network | Function |
| `𝓛` | Loss function | Scalar |
| `∇` | Gradient operator | Operator |
| `𝔼[·]` | Expectation | Operator |
| `𝒩(μ, Σ)` | Gaussian distribution | Distribution |

---

## Appendix B: Hyperparameter Reference

### Default Configuration

```python
# Optimization
learning_rate = 1e-4
warmup_steps = 1000
max_steps = 10000
gradient_clip_norm = 1.0
weight_decay = 1e-5

# Loss weights
w_diffusion = 1.0
w_plddt = 0.01
w_pae = 0.1
w_distogram = 0.01
w_fape = 0.5

# Diffusion
sigma_min = 0.0004
sigma_max = 160.0
sigma_data = 16.0
rho = 7.0

# Data
batch_size = 1
max_num_tokens = 512

# Checkpointing
checkpoint_interval = 1000
keep_n_checkpoints = 5
```

---

## Appendix C: Code Snippet Index

- [Diffusion Loss Implementation](#31-diffusion-loss-primary)
- [pLDDT Loss Implementation](#32-plddt-loss-confidence)
- [Training Loop](#61-training-loop-high-level)
- [Single Training Step](#62-single-training-step-detailed)
- [Data Augmentation](#101-data-augmentation)

---

**End of Report**

For questions or issues, please refer to:
- Main documentation: `TRAINING.md`
- Module README: `src/alphafold3/training/README.md`
- GitHub issues: (repository link)

---

**Report compiled:** November 2025
**Framework version:** 1.0
**AlphaFold 3 base version:** 3.0.1
