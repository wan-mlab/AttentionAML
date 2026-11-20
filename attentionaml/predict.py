"""Apply a trusted saved checkpoint with exact feature alignment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_expression
from .training import load_checkpoint,predict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--expression',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Output exists.')
    model,payload=load_checkpoint(args.checkpoint)
    X,samples,genes=load_expression(args.expression)
    if set(genes)!=set(payload['genes']):
        raise ValueError('Prediction input must have exactly the training feature set; do not silently fill missing genes.')
    index={g:i for i,g in enumerate(genes)}
    X=X[:,[index[g] for g in payload['genes']]]
    classes=payload['classes'];prob=predict(model,X,payload['config'],len(classes),args.device)
    output=pd.DataFrame({'sample_id':samples,'prediction':np.asarray(classes)[prob.argmax(axis=1)]})
    for j,label in enumerate(classes): output['probability::'+label]=prob[:,j]
    args.output.parent.mkdir(parents=True,exist_ok=True);output.to_csv(args.output,index=False)

if __name__=='__main__': main()
