from src.data import get_dataloaders
from src.model import ResNet18
from src.util import compute_f1, compute_precision, compute_recall, plot_structures

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_dataloaders(batch_size = 4, max_length=50, split=0.8, max_data=1000)

# Init model, loss function, optimizer


# Training loop


# Validation loop


# Save model