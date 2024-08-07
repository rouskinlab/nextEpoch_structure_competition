from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures
from submission_formatter import format_submission

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_dataloaders(batch_size = 4, max_length=50, split=0.8, max_data=1000)

# Init model, loss function, optimizer


# Training loop


# Validation loop

# Test loop
structures = []
for sequence in test_loader[1]:
    # Replace with your model prediction !
    structure = (torch.rand(len(sequence), len(sequence))>0.9).type(torch.int) # Has to be shape (L, L) ! 
    structures.append(structure)

format_submission(test_loader[0], test_loader[1], structures, 'test_pred.csv')