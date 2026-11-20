"""Seeded estimators, inference, and complete inference checkpoint payloads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from .models import MLP, MLP_MultiHeadAttention


def build_model(config, n_features, n_classes):
    if config['name']=='MLP':
        return MLP(n_features, config['hidden_size'], n_classes)
    if config['name']=='AttentionAML':
        return MLP_MultiHeadAttention(n_features,config['hidden_size'],n_classes,config['num_heads'])
    raise ValueError('Unknown neural model.')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False


def fit(X,y,config,seed,device='cpu',log_every=25):
    seed_all(seed)
    if config['name']=='RanAML':
        if config['ensemble_size']<1 or config['projection_dim']<1:
            raise ValueError('Positive projection dimension and ensemble size required.')
        rng=np.random.RandomState(seed);m=X.shape[1];p=config['projection_dim'];s=np.sqrt(m)
        members=[]
        for member in range(config['ensemble_size']):
            # Matches original distribution; explicit seeds replace unseeded draws.
            values=rng.choice([np.sqrt(s),0,-np.sqrt(s)],size=(m,p),p=[1/(2*s),1-1/s,1/(2*s)])
            projection=csr_matrix(values,dtype=np.float32)
            del values
            projected=(1/np.sqrt(p))*(projection.T @ X.T).T
            svc=SVC(C=config['C'],kernel=config['kernel'],probability=config['probability'],
                tol=config['tol'],shrinking=config['shrinking'],cache_size=config['cache_size_mb'],
                class_weight=config['class_weight'],max_iter=config['max_iter'],random_state=seed+member)
            svc.fit(projected,y)
            members.append((projection,svc))
            print(f'  projection {member+1}/{config["ensemble_size"]}',flush=True)
        return members,[]
    if config['epochs']<1 or config['batch_size']<1:
        raise ValueError('Positive epochs and batch size required.')
    model=build_model(config,X.shape[1],len(np.unique(y))).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],
        betas=tuple(config['adam_betas']),eps=config['adam_eps'],weight_decay=config['weight_decay'])
    loader=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),
        batch_size=config['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
    model.train();history=[];start=time.perf_counter()
    for epoch in range(config['epochs']):
        total=torch.zeros((),device=device)
        for features,target in loader:
            optimizer.zero_grad()
            result=model(features.to(device));logits=result[0] if isinstance(result,tuple) else result
            loss=nn.functional.cross_entropy(logits,target.to(device))
            loss.backward();optimizer.step()
            total+=loss.detach()*len(target)
        mean_loss=float(total.cpu())/len(y)
        if not np.isfinite(mean_loss): raise RuntimeError('Nonfinite training loss.')
        history.append({'epoch':epoch+1,'loss':mean_loss,'elapsed_seconds':time.perf_counter()-start})
        if (epoch+1)%log_every==0 or epoch==0 or epoch+1==config['epochs']:
            print(f'  epoch {epoch+1}/{config["epochs"]}, loss={mean_loss:.6f}',flush=True)
    return model,history


def predict(estimator,X,config,n_classes,device='cpu'):
    if config['name']=='RanAML':
        probabilities=np.zeros((len(X),n_classes))
        for projection,svc in estimator:
            projected=(1/np.sqrt(config['projection_dim']))*(projection.T @ X.T).T
            probabilities[:,svc.classes_]+=svc.predict_proba(projected)
        result=probabilities/len(estimator)
    else:
        estimator.to(device).eval();blocks=[]
        with torch.no_grad():
            for batch in torch.from_numpy(X).split(256):
                outputs=estimator(batch.to(device));logits=outputs[0] if isinstance(outputs,tuple) else outputs
                blocks.append(torch.softmax(logits,dim=1).cpu().numpy())
        result=np.concatenate(blocks)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1),1,atol=1e-5):
        raise RuntimeError('Invalid prediction probabilities.')
    return result


def save_checkpoint(path,estimator,config,genes,classes,metadata):
    payload=dict(schema_version=1,config=config,genes=genes,classes=classes,metadata=metadata)
    if config['name']=='RanAML':
        import joblib
        payload['members']=estimator;joblib.dump(payload,path,compress=3)
    else:
        payload['state_dict']={k:v.detach().cpu() for k,v in estimator.state_dict().items()}
        torch.save(payload,path)


def load_checkpoint(path):
    if str(path).endswith('.joblib'):
        # joblib is pickle-based: load only artifacts from trusted sources.
        import joblib
        payload=joblib.load(path);estimator=payload['members']
    else:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        estimator=build_model(payload['config'],len(payload['genes']),len(payload['classes']))
        estimator.load_state_dict(payload['state_dict'])
    return estimator,payload
