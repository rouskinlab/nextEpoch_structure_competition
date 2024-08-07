import torch
import json
import csv
import os

def format_submission(references: list, sequences: list, structures: list, save_path: str):
    validate(references, sequences, structures, save_path)

    with open(save_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(['ID', 'Structure'])

        for index, ref in enumerate(references):
            struct = structures[index]
            struct_str = json.dumps(struct.tolist())

            csv_writer.writerow([ref, struct_str])



def validate(references, sequences, structures, save_path):
    params = locals()

    # check that references, sequences, and structures are lists of nonzero length
    for param_name in ['references', 'sequences', 'structures']:
        if not isinstance(params[param_name], list):
            raise TypeError(f"{param_name} must be a list.")
        
        if len(params[param_name]) == 0:
            raise ValueError(f"{param_name} must have length > 0")

    # check that all lists are identical in length
    if not len(references) == len(sequences) == len(structures):
        raise ValueError("The lists 'references', 'sequences', and 'structures' must all have identical length.")

    # check that references and sequences are lists of strings
    for param_name in ['references']:
        if not all(isinstance(value, str) for value in params[param_name]):
            raise TypeError(f"{param_name} must be a list of strings")
        
    # check that structures is list of tensors
    if not all(isinstance(structure, torch.Tensor) for structure in structures):
        raise ValueError("structures must be list of torch tensors.")
    
    for idx, structure in enumerate(structures):
        # check that values of structures are torch tensors
        if not isinstance(structure, torch.Tensor):
            raise ValueError(f"structures must be list of torch tensors, not {type(structure)}.")
        
        # check that structures are square
        if structure.ndimension() != 2 or structure.size(0) != structure.size(1):
            raise ValueError(f"All structures must be square. structure at index {idx} fails condition.")
        
        # checks that structure side length matches sequence length
        if structure.size(0) != len(sequences[idx]):
            raise ValueError(f"structure side length must match sequence length. structure, sequence at index {idx} fail condition. Are your lists ordered properly such that references[i], sequences[i], structures[i] refer to the same RNA?")
        
        # Check that structure are of type integers
        if structure.dtype != torch.int and structure.dtype != torch.long:
            raise ValueError("Structure should be of integer type")
    
    # checks save_path is string
    if not isinstance(save_path, str):
        raise TypeError(f"save_path must be a string, not {type(save_path)}.")