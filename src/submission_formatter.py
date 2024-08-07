import torch
import json
import csv
import os

def format_submission(references: list, sequences: list, matrices: list, save_path: str):
    validate(references, sequences, matrices, save_path)

    with open(save_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(['ID', 'Sequence', 'Structure'])

        for index, ref in enumerate(references):
            seq = sequences[index]
            mtx = matrices[index]
            matrix_str = json.dumps(mtx.tolist())

            csv_writer.writerow([ref, seq, matrix_str])



def validate(references, sequences, matrices, save_path):
    params = locals()

    # check that references, sequences, and matrices are lists of nonzero length
    for param_name in ['references', 'sequences', 'matrices']:
        if not isinstance(params[param_name], list):
            raise TypeError(f"{param_name} must be a list.")
        
        if len(params[param_name]) == 0:
            raise ValueError(f"{param_name} must have length > 0")

    # check that all lists are identical in length
    if not len(references) == len(sequences) == len(matrices):
        raise ValueError("The lists 'references', 'sequences', and 'matrices' must all have identical length.")

    # check that references and sequences are lists of strings
    for param_name in ['references', 'sequences']:
        if not all(isinstance(value, str) for value in params[param_name]):
            raise TypeError(f"{param_name} must be a list of strings")
        
    # check that matrices is list of tensors
    if not all(isinstance(matrix, torch.Tensor) for matrix in matrices):
        raise ValueError("matrices must be list of torch tensors.")
    
    for idx, matrix in enumerate(matrices):
        # check that values of matrices are torch tensors
        if not isinstance(matrix, torch.Tensor):
            raise ValueError(f"matrices must be list of torch tensors, not {type(matrix)}.")
        
        # check that matrices are square
        if matrix.ndimension() != 2 or matrix.size(0) != matrix.size(1):
            raise ValueError(f"All matrices must be square. Matrix at index {idx} fails condition.")
        
        # checks that matrix side length matches sequence length
        if matrix.size(0) != len(sequences[idx]):
            raise ValueError(f"Matrix side length must match sequence length. Matrix, sequence at index {idx} fail condition. Are your lists ordered properly such that references[i], sequences[i], matrices[i] refer to the same RNA?")
    
    # checks save_path is string
    if not isinstance(save_path, str):
        raise TypeError(f"save_path must be a string, not {type(save_path)}.")