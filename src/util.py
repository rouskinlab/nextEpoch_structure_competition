import torch

import matplotlib.pyplot as plt

def compute_f1(pred_matrix, target_matrix, threshold=0.5):
    """
    Compute the F1 score of the predictions.
    :param pred_matrix: Predicted pairing matrix probability  (L,L)
    :param target_matrix: True binary pairing matrix (L,L)
    :return: F1 score for this RNA structure
    """
    pred_matrix = (pred_matrix > threshold).float()
    sum_pair = torch.sum(pred_matrix) + torch.sum(target_matrix)
    if sum_pair == 0:
        return 1.0
    else:
        return (2 * torch.sum(pred_matrix * target_matrix) / sum_pair).item()


def compute_precision(label, pred):
    true_positives = label * pred
    false_positives = (1 - label) * pred
    precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
    return precision.item()

def compute_recall(label, pred):
    true_positives = label * pred
    false_negatives = label * (1 - pred)
    recall = true_positives.sum() / (true_positives.sum() + false_negatives.sum())
    return recall.item()

# def compute_f1(label, pred):
#     precision = compute_precision(label, pred)
#     recall = compute_recall(label, pred)
#     f1 = 2 * precision * recall / (precision + recall)
#     return f1

def plot_structures(file_name: str, pred_matrix, target_matrix, sequence=None, remove_padding=False):

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

    fig.suptitle(f"F1 score = {compute_f1(pred_matrix, target_matrix):.2f}")

    plt.savefig(file_name)
    plt.close(fig)