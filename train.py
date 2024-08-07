from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_dataloaders(batch_size = 4, max_length=50, split=0.8, max_data=1000)

from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures

import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader = get_dataloaders(batch_size=4, max_length=50, split=0.8, max_data=1000)

# Initialize model, loss function, and optimizer
embedding_dim = 50
model = RNA_net(embedding_dim).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        sequences = batch['sequence'].to(device)
        structures = batch['structure'].to(device)

        optimizer.zero_grad()
        outputs, _ = model(sequences)
        loss = criterion(outputs, structures)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Validation loop
model.eval()
with torch.no_grad():
    val_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in val_loader:
        sequences = batch['sequence'].to(device)
        structures = batch['structure'].to(device)

        outputs, _ = model(sequences)
        loss = criterion(outputs, structures)
        val_loss += loss.item()

        preds = torch.sigmoid(outputs).round()
        all_preds.append(preds.cpu())
        all_targets.append(structures.cpu())

    val_loss /= len(val_loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    precision = compute_precision(all_preds, all_targets)
    recall = compute_recall(all_preds, all_targets)
    f1_score = compute_f1(all_preds, all_targets)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

# Save model
model_path = "rna_net_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


# Init model, loss function, optimizer


# Training loop


# Validation loop


# Save model
