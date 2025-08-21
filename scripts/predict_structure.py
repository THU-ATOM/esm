#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import sys
import typing as T
from pathlib import Path
from timeit import default_timer as timer

import torch
import numpy as np

import esm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


PathLike = T.Union[str, Path]


def enable_cpu_offloading(model):
    """Enable CPU offloading for FSDP to reduce GPU memory usage."""
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:9999", world_size=1, rank=0
    )

    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    """Initialize model on GPU with CPU offloading enabled."""
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


def create_batched_sequence_dataset(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    """Create batched sequences to optimize GPU usage."""
    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


def save_confidence_data(output_dir: Path, name: str, output: T.Dict, sequence_length: int):
    """Save confidence data including pLDDT and pTM scores to JSON file."""
    
    # Get pLDDT and atom existence mask, truncated to actual sequence length
    plddt = output["plddt"].cpu().numpy()[:sequence_length]  # Shape: [seq_len, 37]
    atom_exists = output["atom37_atom_exists"].cpu().numpy()[:sequence_length]  # Shape: [seq_len, 37]
    
    # Calculate residue-level pLDDT (average over existing atoms per residue)
    plddt_residue_level = []
    for i in range(sequence_length):  # Only for actual residues
        existing_atoms = atom_exists[i] > 0
        if existing_atoms.any():
            # Average pLDDT over existing atoms for this residue
            residue_plddt = plddt[i][existing_atoms].mean()
        else:
            # If no atoms exist (shouldn't happen), set to 0
            residue_plddt = 0.0
        plddt_residue_level.append(float(residue_plddt))
    
    # Calculate atom-level pLDDT (only for existing atoms, flattened)
    plddt_atom_level = []
    for i in range(sequence_length):
        for j in range(37):
            if atom_exists[i, j] > 0:  # Only include existing atoms
                plddt_atom_level.append(float(plddt[i, j]))
    
    confidence_data = {
        "mean_plddt": float(output["mean_plddt"].cpu().numpy()),
        "ptm": float(output["ptm"].cpu().numpy()),
        "plddt_residue_level": plddt_residue_level,  # [seq_len] - per residue average
        "plddt_atom_level": plddt_atom_level,  # [total_existing_atoms] - flattened list of all existing atoms
    }
    
    # Add predicted aligned error if available
    if "predicted_aligned_error" in output:
        # Also truncate predicted_aligned_error to actual sequence length
        pae = output["predicted_aligned_error"].cpu().numpy()
        if len(pae.shape) == 2:  # Should be [seq_len, seq_len]
            pae_truncated = pae[:sequence_length, :sequence_length]
            confidence_data["predicted_aligned_error"] = pae_truncated.tolist()
        else:
            confidence_data["predicted_aligned_error"] = pae.tolist()
    
    # Add max predicted aligned error if available  
    if "max_predicted_aligned_error" in output:
        confidence_data["max_predicted_aligned_error"] = float(output["max_predicted_aligned_error"].cpu().numpy())
    
    confidence_file = output_dir / f"{name}_confidence.json"
    with open(confidence_file, 'w') as f:
        json.dump(confidence_data, f, indent=2)
    
    return confidence_file


def save_additional_outputs(output_dir: Path, name: str, output: T.Dict, sequence: str):
    """Save additional structural information that might be useful."""
    additional_data = {
        "sequence": sequence,
        "sequence_length": len(sequence),
    }
    
    # Save residue index if available
    if "residue_index" in output:
        additional_data["residue_index"] = output["residue_index"].cpu().numpy().tolist()
    
    # Save chain index if available
    if "chain_index" in output:
        additional_data["chain_index"] = output["chain_index"].cpu().numpy().tolist()
        
    # Save atom37 positions if available (main structural coordinates)
    if "positions" in output:
        additional_data["atom37_positions_shape"] = list(output["positions"].shape)
        # Save the shape information instead of the full coordinates to avoid huge files
        
    # Save atom existence masks
    if "atom37_atom_exists" in output:
        additional_data["atom37_atom_exists"] = output["atom37_atom_exists"].cpu().numpy().tolist()
        
    if "atom14_atom_exists" in output:
        additional_data["atom14_atom_exists"] = output["atom14_atom_exists"].cpu().numpy().tolist()
    
    additional_file = output_dir / f"{name}_additional.json"
    with open(additional_file, 'w') as f:
        json.dump(additional_data, f, indent=2)
    
    return additional_file


def create_parser():
    """Create argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Predict protein structures from JSON input using ESMFold"
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input JSON file containing list of {'name': str, 'sequence': str} dicts",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output directory where results will be saved",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        help="Parent path to pretrained ESM data directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycles to run. Defaults to number used in training (4).",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help="Maximum number of tokens per GPU forward-pass for batched prediction.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for axial attention computation to reduce memory usage.",
    )
    parser.add_argument("--cpu-only", help="Run on CPU only", action="store_true")
    parser.add_argument(
        "--cpu-offload", help="Enable CPU offloading for memory efficiency", action="store_true"
    )
    parser.add_argument(
        "--save-additional",
        help="Save additional structural information to JSON files",
        action="store_true",
    )
    return parser


def load_input_json(input_file: Path) -> T.List[T.Dict[str, str]]:
    """Load and validate input JSON file."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of dictionaries")
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dictionary")
        if "name" not in item or "sequence" not in item:
            raise ValueError(f"Item {i} missing required 'name' or 'sequence' field")
        if not isinstance(item["name"], str) or not isinstance(item["sequence"], str):
            raise ValueError(f"Item {i} 'name' and 'sequence' must be strings")
    
    return data


def run(args):
    """Main execution function."""
    # Load input data
    logger.info(f"Loading sequences from {args.input}")
    input_data = load_input_json(args.input)
    logger.info(f"Loaded {len(input_data)} sequences from {args.input}")
    
    # Create output directory
    args.output.mkdir(exist_ok=True, parents=True)
    
    # Prepare sequences for processing
    all_sequences = [(item["name"], item["sequence"]) for item in input_data]
    
    # Sort by sequence length for more efficient batching
    all_sequences = sorted(all_sequences, key=lambda header_seq: len(header_seq[1]))
    
    # Load ESMFold model
    logger.info("Loading ESMFold model")
    if args.model_dir is not None:
        torch.hub.set_dir(args.model_dir)
    
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    model.set_chunk_size(args.chunk_size)
    
    # Configure device and memory settings
    if args.cpu_only:
        model.esm.float()  # Convert to fp32 as ESM-2 in fp16 is not supported on CPU
        model.cpu()
        logger.info("Running on CPU only")
    elif args.cpu_offload:
        model = init_model_on_gpu_with_cpu_offloading(model)
        logger.info("Running on GPU with CPU offloading")
    else:
        model.cuda()
        logger.info("Running on GPU")
    
    logger.info("Starting structure predictions")
    batched_sequences = create_batched_sequence_dataset(all_sequences, args.max_tokens_per_batch)
    
    num_completed = 0
    num_sequences = len(all_sequences)
    
    for headers, sequences in batched_sequences:
        start = timer()
        try:
            output = model.infer(sequences, num_recycles=args.num_recycles)
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                if len(sequences) > 1:
                    logger.error(
                        f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                        "Try lowering `--max-tokens-per-batch`."
                    )
                else:
                    logger.error(
                        f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}."
                    )
                continue
            raise
        
        # Move output to CPU to free GPU memory
        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        
        total_time = timer() - start
        time_string = f"{total_time / len(headers):0.1f}s"
        if len(sequences) > 1:
            time_string = time_string + f" (amortized, batch size {len(sequences)})"
        
        # Process each sequence in the batch
        for i, (header, seq, pdb_string, mean_plddt, ptm) in enumerate(zip(
            headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
        )):
            # Create individual output directory for each sequence
            seq_output_dir = args.output / header
            seq_output_dir.mkdir(exist_ok=True)
            
            # Save PDB file
            pdb_file = seq_output_dir / f"{header}.pdb"
            pdb_file.write_text(pdb_string)
            
            # Extract individual outputs for this sequence
            individual_output = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor) and len(value.shape) > 0 and value.shape[0] > i:
                    individual_output[key] = value[i]
                else:
                    individual_output[key] = value
            
            # Save confidence data
            confidence_file = save_confidence_data(seq_output_dir, header, individual_output, len(seq))
            
            # Save additional structural information if requested
            if args.save_additional:
                additional_file = save_additional_outputs(seq_output_dir, header, individual_output, seq)
                logger.info(f"Saved additional data to {additional_file}")
            
            num_completed += 1
            logger.info(
                f"Predicted structure for {header} (length {len(seq)}) - "
                f"pLDDT: {mean_plddt:0.1f}, pTM: {ptm:0.3f} in {time_string}. "
                f"Saved to {seq_output_dir}. "
                f"Progress: {num_completed}/{num_sequences}"
            )
    
    logger.info(f"Completed all {num_sequences} structure predictions!")
    logger.info(f"Results saved to: {args.output}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
