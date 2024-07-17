import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import math
import os, pickle
import torch
from torch.utils.data import Dataset, DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_predict(tpm_test):
    test = torch.tensor(tpm_test.values.astype('float32'), dtype=torch.float32)
    test_loader = DataLoader(test,shuffle=False)
    pred = []
    model = torch.load(os.path.dirname(__file__) + '/model/MLP_attention.pth')
    for data in test_loader:
        outputs = model(data.to(device))   
        _, predicted = torch.max(outputs, 1)
        pred.append(predicted.cpu().numpy()[0])
    label_transform = {0:'BCL11B', 1:'CBFB-GDXY', 2:'CBFB::MYH11',3:'CEBPA', 4:'DEK::NUP214',
       5:'ETS family', 6:'GATA1', 7:'GLISr',
       8:'HOXr', 9:'KAT6Ar', 10:'KMT2A-PTD', 11:'KMT2Ar',
       12:'MECOM', 13:'MNX1', 14:'NPM1', 15:'NUP98r', 16:'PICALM::MLLT10',
       17:'RBM15::MKL1', 18:'RUNX1::RUNX1T1', 19:'RUNX1::RUNX1T1-like',20:'UBTF'}
    sub_pred = [label_transform[value] for value in pred]

    results = {'Samples':tpm_test.index, 'Subtype_pred': sub_pred}
    results = pd.DataFrame(results)
    results.to_csv('Prediction_results.csv', index=False)
    print(results)
    return results
