"""Neural architectures extracted verbatim from the supplied source scripts."""
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        # Split into heads
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Concatenate heads and pass through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        return self.fc_out(output)

class MLP_MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_heads=8):
        super(MLP_MultiHeadAttention, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.attention1 = MultiHeadAttention(hidden_size, num_heads)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.attention2 = MultiHeadAttention(hidden_size // 2, num_heads // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        # Add a sequence dimension if x is just (batch_size, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.relu(self.layer1(x))
        x = x + self.attention1(x)  # Residual connection
        x = self.layer_norm1(x)
        
        x = self.relu(self.layer2(x))
        x = x + self.attention2(x)  # Residual connection
        x = self.layer_norm2(x)
        
        x = self.layer3(x)
        return x.squeeze(1),F.softmax(x, dim=-1)
