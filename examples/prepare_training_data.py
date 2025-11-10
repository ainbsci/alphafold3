#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0.

"""Example script to prepare training data from PDB files.

This script demonstrates how to organize your data for AlphaFold 3 training.

Usage:
  python prepare_training_data.py \
    --pdb_dir=/path/to/pdb/files \
    --output_dir=/path/to/training/data
"""

import pathlib
import shutil
from absl import app
from absl import flags

flags.DEFINE_string('pdb_dir', None, 'Directory containing PDB/mmCIF files')
flags.DEFINE_string('output_dir', None, 'Output directory for training data')
FLAGS = flags.FLAGS


def main(_):
  if not FLAGS.pdb_dir or not FLAGS.output_dir:
    print('Error: --pdb_dir and --output_dir are required')
    return

  pdb_dir = pathlib.Path(FLAGS.pdb_dir)
  output_dir = pathlib.Path(FLAGS.output_dir)

  # Create output structure
  structures_dir = output_dir / 'structures'
  structures_dir.mkdir(parents=True, exist_ok=True)

  # Copy PDB/mmCIF files to structures directory
  pdb_files = list(pdb_dir.glob('*.pdb')) + list(pdb_dir.glob('*.cif'))

  print(f'Found {len(pdb_files)} structure files')

  for pdb_file in pdb_files:
    # Convert PDB to mmCIF if needed (simplified - in practice use BioPython)
    if pdb_file.suffix == '.pdb':
      # For this example, just copy as-is
      # In practice, convert PDB to mmCIF format
      dest = structures_dir / f'{pdb_file.stem}.cif'
      print(f'Note: {pdb_file.name} should be converted to mmCIF format')
    else:
      dest = structures_dir / pdb_file.name

    shutil.copy(pdb_file, dest)
    print(f'Copied: {pdb_file.name} -> {dest}')

  print(f'\nData preparation complete!')
  print(f'Training data organized in: {output_dir}')
  print(f'\nYou can now train with:')
  print(f'  python train_alphafold.py \\')
  print(f'    --train_data_dir={output_dir} \\')
  print(f'    --model_dir=/path/to/pretrained/model \\')
  print(f'    --output_dir=/path/to/output')


if __name__ == '__main__':
  flags.mark_flags_as_required(['pdb_dir', 'output_dir'])
  app.run(main)
