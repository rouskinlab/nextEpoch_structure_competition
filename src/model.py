import torch
import torch.nn as nn

class RNA_embedding(nn.Module):

    '''
    This class is a simple embedding layer for RNA sequences. 

    input:
    - embedding_dim: int, dimension of the embedding space
    - vocab_size: int, size of the vocabulary (number of different nucleotides)

    output:
    - x: tensor, (N, embedding_dim, L, L), where N is the batch size, L is the length of the sequence 
    '''

    def __init__(self, embedding_dim, vocab_size=5):
        super(RNA_embedding, self).__init__()

        self.table_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_input = nn.Linear(embedding_dim*2, embedding_dim)

    def forward(self, x): # x is (N, L) -> embedded as sequence of integer

        # Sequence representation
        s = self.table_embedding(x)                         # (N, L, embedding_dim)

        # Outer concatenation to get matrix representation
        m = s.unsqueeze(2).repeat(1, 1, s.shape[1], 1)      # (N, L, L, embedding_dim)
        m = torch.cat((m, m.permute(0, 2, 1, 3)), dim=-1)   # (N, L, L, 2*embedding_dim)

        # Bring back to embedding dimension
        m = self.fc_input(m)                                # (N, L, L, embedding_dim)    

        m = m.permute(0, 3, 1, 2)                           # (N, embedding_dim , L, L)   

        return s, m

class RNA_net(nn.Module):
    def __init__(self, embedding_dim):
        super(RNA_net, self).__init__()

        # Define the embedding layer
        self.embedding = RNA_embedding(embedding_dim)
        
        # Define your layers for the network
        self.conv1 = nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (embedding_dim // 2) * (embedding_dim // 2), 512)
        self.fc2 = nn.Linear(512, 1)  # Assuming binary classification for secondary structure

    def forward(self, x):
        # Get embeddings
        s, m = self.embedding(x)
        
        # Apply convolutional layers
        x = self.pool(F.relu(self.conv1(m)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Example usage
# Define input size and create the model
embedding_dim = 50
model = RNA_net(embedding_dim)
print(model)

# This is the class to be developed !
#class RNA_net(nn.Module):

#   def __init__(self, embedding_dim):
#      super(RNA_net, self).__init__()

#        self.embedding = RNA_embedding(embedding_dim)

#       # Your layers here

#    def forward(self, x):

#       s, m = self.embedding(x)

#        # Your forward pass here

#       return m
