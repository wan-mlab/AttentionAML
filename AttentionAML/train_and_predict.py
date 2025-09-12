import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import math
import os, pickle
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn
import torch.nn.functional as F
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
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.attention2 = MultiHeadAttention(hidden_size // 2, num_heads // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        #self.layer_norm1 = nn.LayerNorm(hidden_size)
        #self.layer_norm2 = nn.LayerNorm(hidden_size // 2)

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
        
def train_and_predict(tpm_test):
    test = torch.tensor(tpm_test.values.astype('float32'), dtype=torch.float32)
    test_loader = DataLoader(test,shuffle=False)
    pred = []
    pro_list = []
    model = MLP_MultiHeadAttention(input_size=tpm_test.shape[1], hidden_size=512, num_classes=21, num_heads=8).to(device)
    model.load_state_dict(torch.load(os.path.dirname(__file__) + '/model/MLP_attention.pth'))
    model.eval() 

    with torch.no_grad():
        for data in test_loader:
            outputs,pro = model(data.to(device))
            _, predicted = torch.max(outputs, 1)
            pro_list.append(pro.cpu().reshape(21))
            predicted_labels.append(predicted.cpu())

    stacked_tensor = torch.stack(pro_list)
    numpy_array = stacked_tensor.numpy()
    numpy_array = pd.DataFrame(numpy_array)
    prob = numpy_array.max(axis=1)
    pred = torch.cat(predicted_labels).numpy()

    label_transform = {0:'BCL11B', 1:'CBFB-GDXY', 2:'CBFB::MYH11',3:'CEBPA', 4:'DEK::NUP214',
       5:'ETS family', 6:'GATA1', 7:'GLISr',
       8:'HOXr', 9:'KAT6Ar', 10:'KMT2A-PTD', 11:'KMT2Ar',
       12:'MECOM', 13:'MNX1', 14:'NPM1', 15:'NUP98r', 16:'PICALM::MLLT10',
       17:'RBM15::MKL1', 18:'RUNX1::RUNX1T1', 19:'RUNX1::RUNX1T1-like',20:'UBTF'}
    sub_pred = [label_transform[value] for value in pred]

    results = {'Samples':tpm_test.index, 'Subtype_pred': sub_pred, 'Probabolity': prob}
    results = pd.DataFrame(results)
    results.to_csv('Prediction_results.csv', index=False)
    print(results)
    return results




