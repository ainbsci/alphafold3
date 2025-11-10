# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Data loading utilities for AlphaFold 3 training."""

from typing import Iterator, Optional, Dict, Any
import pathlib
import json

import numpy as np
from alphafold3 import structure as af3_structure
from alphafold3.common import folding_input
from alphafold3.data import featurisation
from alphafold3.model import features
from alphafold3.constants import chemical_components


class StructureDataset:
  """Dataset for loading protein structures for training.

  This dataset loads structures from mmCIF files or AlphaFold3 JSON inputs
  and prepares them for training.
  """

  def __init__(
      self,
      data_dir: str,
      max_num_tokens: int = 512,
      use_cache: bool = True,
  ):
    """Initialize the dataset.

    Args:
      data_dir: Directory containing training data
        Expected structure:
          data_dir/
            structures/  # mmCIF or PDB files
            inputs/      # AlphaFold3 JSON input files
      max_num_tokens: Maximum number of tokens per sample
      use_cache: Whether to cache featurized examples
    """
    self.data_dir = pathlib.Path(data_dir)
    self.max_num_tokens = max_num_tokens
    self.use_cache = use_cache

    # Find all structure or input files
    self.structure_files = list((self.data_dir / 'structures').glob('*.cif'))
    self.structure_files += list((self.data_dir / 'structures').glob('*.pdb'))
    self.input_files = list((self.data_dir / 'inputs').glob('*.json'))

    if not self.structure_files and not self.input_files:
      raise ValueError(
          f'No structure or input files found in {self.data_dir}'
      )

    self._cache = {} if use_cache else None

  def __len__(self) -> int:
    """Number of samples in dataset."""
    return len(self.structure_files) + len(self.input_files)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    """Get a single training example.

    Args:
      idx: Index of the example

    Returns:
      Dictionary containing:
        - batch: Featurized input batch
        - true_positions: Ground truth atom positions
        - structure: Original structure object
    """
    # Check cache
    if self._cache is not None and idx in self._cache:
      return self._cache[idx]

    # Load data
    if idx < len(self.structure_files):
      example = self._load_from_structure(self.structure_files[idx])
    else:
      json_idx = idx - len(self.structure_files)
      example = self._load_from_json(self.input_files[json_idx])

    # Cache if enabled
    if self._cache is not None:
      self._cache[idx] = example

    return example

  def _load_from_structure(
      self, structure_file: pathlib.Path
  ) -> Dict[str, Any]:
    """Load training example from a structure file (mmCIF or PDB).

    Args:
      structure_file: Path to structure file

    Returns:
      Training example dictionary
    """
    # Load structure
    struc = af3_structure.from_mmcif(
        mmcif_string=structure_file.read_text(),
        name=structure_file.stem,
    )

    # Extract ground truth positions
    true_positions = np.stack(
        [struc.atom_x, struc.atom_y, struc.atom_z], axis=-1
    )

    # Convert structure to AlphaFold3 input format
    fold_input = self._structure_to_fold_input(struc)

    # Featurize
    ccd = chemical_components.cached_ccd()
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=[self.max_num_tokens],
        ccd=ccd,
        verbose=False,
    )

    # Get first (and only) example
    batch = featurised_examples[0]

    return {
        'batch': batch,
        'true_positions': true_positions,
        'structure': struc,
        'name': structure_file.stem,
    }

  def _load_from_json(self, json_file: pathlib.Path) -> Dict[str, Any]:
    """Load training example from AlphaFold3 JSON input file.

    Args:
      json_file: Path to JSON file

    Returns:
      Training example dictionary
    """
    # Load JSON input
    with open(json_file) as f:
      json_data = json.load(f)

    fold_input = folding_input.Input.from_json(json.dumps(json_data))

    # Load ground truth structure if available
    structure_file = json_file.parent.parent / 'structures' / f'{json_file.stem}.cif'
    if structure_file.exists():
      struc = af3_structure.from_mmcif(
          mmcif_string=structure_file.read_text(),
          name=json_file.stem,
      )
      true_positions = np.stack(
          [struc.atom_x, struc.atom_y, struc.atom_z], axis=-1
      )
    else:
      # No ground truth available - use zeros as placeholder
      # (This case shouldn't happen in training but is here for robustness)
      struc = None
      true_positions = None

    # Featurize
    ccd = chemical_components.cached_ccd()
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=[self.max_num_tokens],
        ccd=ccd,
        verbose=False,
    )

    batch = featurised_examples[0]

    return {
        'batch': batch,
        'true_positions': true_positions,
        'structure': struc,
        'name': json_file.stem,
    }

  def _structure_to_fold_input(
      self, struc: af3_structure.Structure
  ) -> folding_input.Input:
    """Convert a Structure object to a FoldInput.

    This creates a minimal FoldInput for training purposes.

    Args:
      struc: Structure object

    Returns:
      FoldInput object
    """
    # Extract chains and sequences
    chains = []
    unique_chain_ids = sorted(set(struc.chains))

    for chain_id in unique_chain_ids:
      # Get residues for this chain
      chain_mask = np.array(struc.chains) == chain_id
      chain_residues = [
          struc.res_names[i] for i in range(len(struc.res_names))
          if chain_mask[i]
      ]

      # Convert to sequence (simplified - assumes standard amino acids)
      sequence = self._residues_to_sequence(chain_residues)

      if sequence:
        chains.append(
            folding_input.ProteinChain(
                id=chain_id,
                sequence=sequence,
            )
        )

    # Create fold input
    fold_input = folding_input.Input(
        name=struc.name,
        chains=chains,
        rng_seeds=[0],  # Dummy seed for training
        user_ccd=None,
    )

    return fold_input

  def _residues_to_sequence(self, residues: list[str]) -> str:
    """Convert residue names to single-letter amino acid sequence.

    Args:
      residues: List of 3-letter residue codes

    Returns:
      Single-letter amino acid sequence
    """
    # Standard amino acid mapping
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    sequence = ''
    for res in residues:
      # Handle modified residues by using the base residue
      base_res = res[:3].upper()
      if base_res in aa_map:
        sequence += aa_map[base_res]
      # Skip non-standard residues (ligands, etc.)

    return sequence


def create_dataloader(
    dataset: StructureDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Iterator[Dict[str, Any]]:
  """Create a data loader for training.

  Args:
    dataset: StructureDataset instance
    batch_size: Batch size (currently only supports 1 due to variable sizes)
    shuffle: Whether to shuffle data
    num_workers: Number of workers for data loading

  Yields:
    Batches of training examples
  """
  # Create indices
  indices = list(range(len(dataset)))

  if shuffle:
    np.random.shuffle(indices)

  # Simple sequential loading (batch_size=1 for now due to variable sizes)
  for idx in indices:
    yield dataset[idx]


def collate_fn(examples: list[Dict[str, Any]]) -> Dict[str, Any]:
  """Collate function for batching examples.

  Currently only supports batch_size=1 due to variable sequence lengths.

  Args:
    examples: List of examples

  Returns:
    Batched example
  """
  # For now, just return single example
  # In practice, you'd need padding/bucketing for batching
  return examples[0]
