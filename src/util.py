import torch

import matplotlib.pyplot as plt

def compute_f1(pred_matrix, target_matrix, threshold=0.5):
    
    """
    Compute the F1 score of the predictions.

    args:
    - pred_matrix: Predicted pairing matrix probability  (L,L)
    - target_matrix: True binary pairing matrix (L,L)
    - threshold: float, threshold to binarize the predicted matrix

    return:
    - f1: float, F1 score
    """
    pred_matrix = (pred_matrix > threshold).float()
    sum_pair = torch.sum(pred_matrix) + torch.sum(target_matrix)
    if sum_pair == 0:
        return 1.0
    else:
        return (2 * torch.sum(pred_matrix * target_matrix) / sum_pair).item()


def compute_precision(pred_matrix, target_matrix, threshold=0.5):

    """
    Compute the Precision score of the predictions.

    args:
    - pred_matrix: Predicted pairing matrix probability  (L,L)
    - target_matrix: True binary pairing matrix (L,L)
    - threshold: float, threshold to binarize the predicted matrix

    return:
    - precision: float, Precision score
    """

    pred_matrix = (pred_matrix > threshold).float()

    true_positives = target_matrix * pred_matrix
    false_positives = (1 - target_matrix) * pred_matrix
    precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
    return precision.item()

def compute_recall(pred_matrix, target_matrix, threshold=0.5):

    """
    Compute the Recall score of the predictions.

    args:
    - pred_matrix: Predicted pairing matrix probability  (L,L)
    - target_matrix: True binary pairing matrix (L,L)
    - threshold: float, threshold to binarize the predicted matrix

    return:
    - recall: float, Recall score
    """

    pred_matrix = (pred_matrix > threshold).float()

    true_positives = target_matrix * pred_matrix
    false_negatives = target_matrix * (1 - pred_matrix)
    recall = true_positives.sum() / (true_positives.sum() + false_negatives.sum())
    return recall.item()


def plot_structures(file_name: str, pred_matrix, target_matrix, sequence=None, remove_padding=False):

    """
    Plot the predicted and target pairing matrices side by side.

    args:
    - file_name: str, file name to save the plot
    - pred_matrix: tensor, (L,L), predicted pairing matrix
    - target_matrix: tensor, (L,L), target pairing matrix
    - sequence: tensor, (L,), RNA sequence
    - remove_padding: bool, whether to remove padding from the sequence
    """

    if remove_padding:
        assert sequence is not None, "Need to provide sequence to remove padding"
        seq_len = torch.count_nonzero(sequence).item()

        pred_matrix = pred_matrix[:seq_len, :seq_len]
        target_matrix = target_matrix[:seq_len, :seq_len]


    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(pred_matrix, vmin=0, vmax=1, cmap='viridis')
    ax2.imshow(target_matrix, vmin=0, vmax=1, cmap='viridis')

    # add titles
    ax1.title.set_text('Predicted Structure')
    ax2.title.set_text('Target Structure')

    fig.suptitle(f"Precision = {compute_precision(pred_matrix, target_matrix):.2f}, Recall = {compute_recall(pred_matrix, target_matrix):.2f}, F1 score = {compute_f1(pred_matrix, target_matrix):.2f}")

    plt.savefig(file_name)
    plt.close(fig)