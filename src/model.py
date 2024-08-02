import torch
import torch.nn as nn

class RNA_embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(RNA_embedding, self).__init__()

        self.table_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_input = nn.Linear(embedding_dim*2, embedding_dim)

    def forward(self, x): # x is (N, L) -> embedded as sequence of integer

        x = self.table_embedding(x)                         # (N, L, embedding_dim)

        # Outer concatenation
        x = x.unsqueeze(2).repeat(1, 1, x.shape[1], 1)      # (N, L, L, embedding_dim)
        x = torch.cat((x, x.permute(0, 2, 1, 3)), dim=-1)   # (N, L, L, 2*embedding_dim)

        # Bring back to embedding dimension
        x = self.fc_input(x)                                # (N, L, L, embedding_dim)        

        return x

class RNA_net(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(RNA_net, self).__init__()

        self.embedding = RNA_embedding(vocab_size, embedding_dim)
        

        # Your layers here

    def forward(self, x):

        x = self.embedding(x)

        # Your forward pass here

        return x
