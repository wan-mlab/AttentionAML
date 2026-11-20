"""Shared classification metrics for held-out evaluation."""
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef

def metrics(y,pred):
    return dict(Accuracy=accuracy_score(y,pred),Precision=precision_score(y,pred,average='weighted',zero_division=0),
        Recall=recall_score(y,pred,average='weighted',zero_division=0),F1=f1_score(y,pred,average='weighted',zero_division=0),
        MCC=matthews_corrcoef(y,pred),MacroF1=f1_score(y,pred,average='macro',zero_division=0))

