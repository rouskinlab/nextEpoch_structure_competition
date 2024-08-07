from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures
from src.submission_formatter import format_submission

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_dataloaders(batch_size = 2, max_length=50, split=0.8, max_data=1000)

# Init model, loss function, optimizer
model = RNA_net(embedding_dim=32)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):

    loss_epoch = 0.0
    metric_epoch = 0.0

    for batch in train_loader:

        x = batch["sequence"] # (N, L)
        y = batch['structure'] # (N, L, L)

        y_pred = model(x)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loss_epoch += loss.item()
        metric_epoch += compute_f1(y_pred, y)

    print("Loss", loss_epoch/len(train_loader))
    print("F1 score", metric_epoch/len(train_loader))


# Validation loop

# Test loop
structures = []
for sequence in test_loader[1]:
    # Replace with your model prediction !
    structure = (torch.rand(len(sequence), len(sequence))>0.9).type(torch.int) # Has to be shape (L, L) ! 
    structures.append(structure)

format_submission(test_loader[0], test_loader[1], structures, 'test_pred.csv')