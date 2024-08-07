import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import pandas as pd

import os

seq2int = {'N' : 0, 'A': 1, 'C': 2, 'G': 3, 'U': 4}


class RNADataset(Dataset):
    def __init__(self, json_file, transform=None, max_length=None, max_data=None):
        with open(json_file, 'r') as f:
            self.rna_data = json.load(f)

        # Filter out sequences longer than max_length
        self.rna_data = pd.read_json(json_file).T

        if not max_length: max_length = self.rna_data['sequence'].apply(len).max()
        self.rna_data = self.rna_data[self.rna_data['sequence'].apply(len) <= max_length]

        if max_data:
            if len(self.rna_data)>max_data: self.rna_data = self.rna_data.sample(max_data)
            else: print(f"Cannot get {max_data} datapoints from dataset, only got {len(self.rna_data)}. Returning maximum.")

        self.transform = transform
        self.max_length = max_length # Add this line

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence = self.rna_data.iloc[idx]['sequence']
        structure = self.rna_data.iloc[idx]['structure']
        reference = self.rna_data.index[idx]

        sequence += 'N' * (self.max_length - len(sequence)) # Add padding
        sample = {'reference': reference, 'sequence': sequence, 'structure': structure}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Define the transformation class
class ToTensor(object):

    def pairs_to_matrix(self, pairs, length):
        matrix = torch.zeros(length, length, dtype=torch.float)
        pairs = torch.tensor(pairs)
        
        if len(pairs) > 0:
            matrix[pairs[:, 0], pairs[:, 1]] = 1
            matrix[pairs[:, 1], pairs[:, 0]] = 1

        return matrix
    
    def __call__(self, sample):

        sequence, structure = sample['sequence'], sample['structure']

        # Replace unknown characters with a valid character ('A' in this case)
        sequence_tensor = torch.tensor([seq2int[c] for c in sequence], dtype=torch.long)
        structure_tensor = self.pairs_to_matrix(structure, len(sequence))

        return {'reference': sample['reference'], 'sequence': sequence_tensor, 'structure': structure_tensor}



def get_dataloaders(batch_size=32, max_length=100, split=0.8, max_data=None):

    '''
    This function returns the training and validation dataloaders.
    
    input:
    - batch_size: int, size of the batch
    - max_length: int, maximum length of the RNA sequences
    - split: float, fraction of the data to be used for training
    - max_data: int, maximum number of data points to be used

    output:
    - train_loader: DataLoader, training dataloader
    - val_loader: DataLoader, validation dataloader
    '''

    assert split > 0 and split < 1, 'Split must be between 0 and 1'
    assert max_length > 0, 'Max length must be greater than 0'
    assert batch_size > 0, 'Batch size must be greater than 0'

    current_dir = os.path.dirname(os.path.realpath(__file__))

    trainValidation_dataset = RNADataset(os.path.join(current_dir, 'data/data_train_rnastralign_archiveII.json'), transform=ToTensor(), max_length=max_length, max_data=max_data)

    assert len(trainValidation_dataset) > 0, 'Not enough training data found'

    # Split the dataset into training and validation sets
    train_size = int(split * len(trainValidation_dataset))
    val_size = len(trainValidation_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainValidation_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get test set
    data_test = pd.read_json(os.path.join(current_dir, 'data/test.json')).T
    test_references = data_test.index.tolist()
    test_sequences = [torch.tensor([seq2int[c] for c in sequence], dtype=torch.long) for sequence in data_test.sequence.tolist()]

    return train_loader, val_loader, (test_references, test_sequences)



    
