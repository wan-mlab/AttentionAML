"""Shared stratified 80/20 holdout for MLP, RanAML and AttentionAML."""
import argparse
import gc
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .data import load_data,sha256,write_json
from .training import fit,predict,save_checkpoint
from .metrics import metrics
from .holdout_plots import plot_results
import torch

ROOT=Path(__file__).resolve().parents[1]


def make_split(samples,y,seed,split_file=None):
    if split_file:
        table=pd.read_csv(split_file,dtype=str)
        if not {'sample_id','split'}<=set(table) or table[['sample_id','split']].isna().any().any():
            raise ValueError('Split CSV requires nonmissing sample_id,split columns.')
        if table.sample_id.duplicated().any() or set(table.sample_id)!=set(samples):
            raise ValueError('Split CSV must cover every included sample exactly once.')
        labels=table.set_index('sample_id').loc[samples,'split'].to_numpy()
        if set(labels)!= {'train','test'}: raise ValueError('Split values must be train and test.')
        train=np.flatnonzero(labels=='train');test=np.flatnonzero(labels=='test')
    else:
        train,test=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=seed)
    if set(y[train])!=set(np.unique(y)):
        raise ValueError('Training partition lacks a subtype; cannot train this closed-set comparison.')
    return np.sort(train),np.sort(test)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expression',type=Path)
    parser.add_argument('--labels',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--models',nargs='+',choices=['MLP','RanAML','AttentionAML'],default=['MLP','RanAML','AttentionAML'])
    parser.add_argument('--config-dir',type=Path,default=ROOT/'configs')
    parser.add_argument('--seed',type=int,default=42,help='Split seed, fixed before evaluating results')
    parser.add_argument('--training-seed',type=int,default=42)
    parser.add_argument('--split-file',type=Path,help='Reuse an existing sample_id,split CSV instead of generating a split')
    parser.add_argument('--exclude',nargs='*',default=['Unclassified','APL','BCR::ABL1'])
    parser.add_argument('--patient-map',type=Path,help='Optional sample_id,patient_id CSV to detect cross-partition patient overlap')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--log-every',type=int,default=25)
    parser.add_argument('--no-checkpoints',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Audit split without training')
    parser.add_argument('--plot-only',action='store_true',help='Regenerate figures from existing output predictions')
    args=parser.parse_args()
    if args.plot_only:
        run=json.loads((args.output/'run.json').read_text())
        plot_results(args.output,run['classes'],run['models']);return
    if args.expression is None or args.labels is None: parser.error('--expression and --labels are required')
    if args.threads<1 or args.log_every<1: parser.error('threads and log-every must be positive')
    if len(args.models)!=len(set(args.models)): parser.error('Duplicate models')
    if args.output.exists(): parser.error('Output exists; use a fresh directory or --plot-only')
    if args.device.startswith('cuda') and not torch.cuda.is_available(): parser.error('CUDA is unavailable')
    torch.set_num_threads(args.threads)
    X,y,samples,genes,classes=load_data(args.expression,args.labels,args.exclude)
    train,test=make_split(samples,y,args.seed,args.split_file)
    if args.patient_map:
        table=pd.read_csv(args.patient_map,dtype=str)
        if not {'sample_id','patient_id'}<=set(table) or table.sample_id.duplicated().any():
            raise ValueError('Patient map requires unique sample_id and patient_id columns.')
        patient=table.set_index('sample_id').reindex(samples).patient_id
        if patient.isna().any(): raise ValueError('Missing patient IDs')
        if set(patient.iloc[train]) & set(patient.iloc[test]):
            raise ValueError('Patient overlap between train and test. Supply a predefined patient-disjoint --split-file.')
    config=[json.loads((args.config_dir/f'{name}.json').read_text()) for name in args.models]
    if [c['name'] for c in config]!=args.models: raise ValueError('Model config name mismatch')
    args.output.mkdir(parents=True)
    split=pd.DataFrame({'sample_id':samples,'category':np.asarray(classes)[y],'split':'train'})
    split.loc[test,'split']='test';split.to_csv(args.output/'split.csv',index=False)
    counts=pd.crosstab(split.category,split.split).reindex(index=classes,columns=['train','test'],fill_value=0)
    counts.to_csv(args.output/'subtype_counts.csv');print(counts.to_string(),flush=True)
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','joblib','matplotlib']}
    manifest=dict(models=args.models,configs=config,classes=classes,genes=genes,
        n_train=len(train),n_test=len(test),actual_test_fraction=len(test)/len(y),
        split_seed=args.seed,training_seed=args.training_seed,split_method='provided' if args.split_file else 'stratified_80_20',
        split_file_sha256=sha256(args.split_file) if args.split_file else None,
        patient_overlap_checked=args.patient_map is not None,excluded_categories=args.exclude,
        expression_sha256=sha256(args.expression),labels_sha256=sha256(args.labels),
        split_sha256=sha256(args.output/'split.csv'),device=args.device,threads=args.threads,
        python=platform.python_version(),versions=versions,dry_run=args.dry_run)
    write_json(args.output/'run.json',manifest)
    print(f'Train: {len(train)}; held-out test: {len(test)} ({len(test)/len(y):.2%}).',flush=True)
    if args.dry_run:return
    rows=[];completed=[]
    for cfg in config:
        name=cfg['name'];directory=args.output/name;directory.mkdir()
        print(f'Training {name} on training partition only; device={args.device}',flush=True)
        start=time.perf_counter()
        model,history=fit(X[train],y[train],cfg,args.training_seed,args.device,args.log_every)
        probability=predict(model,X[test],cfg,len(classes),args.device)
        pred=probability.argmax(axis=1)
        frame=pd.DataFrame({'sample_id':np.asarray(samples)[test],'true_label':np.asarray(classes)[y[test]],
            'predicted_label':np.asarray(classes)[pred],'correct':pred==y[test],
            'predicted_probability':probability.max(axis=1),
            'true_class_probability':probability[np.arange(len(test)),y[test]]})
        for j,label in enumerate(classes):frame['probability::'+label]=probability[:,j]
        frame.to_csv(directory/'test_predictions.csv',index=False)
        report=classification_report(y[test],pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
        pd.DataFrame(report).T.to_csv(directory/'classification_report.csv')
        if history:pd.DataFrame(history).to_csv(directory/'training_history.csv',index=False)
        if not args.no_checkpoints:
            suffix='joblib' if name=='RanAML' else 'pt'
            save_checkpoint(directory/f'model.{suffix}',model,cfg,genes,classes,
                dict(training_seed=args.training_seed,n_train=len(train),split_sha256=manifest['split_sha256'],
                     expression_sha256=manifest['expression_sha256'],labels_sha256=manifest['labels_sha256'],versions=versions))
        rows.append(dict(model=name,n_train=len(train),n_test=len(test),seconds=time.perf_counter()-start,
                         **metrics(y[test],pred)))
        pd.DataFrame(rows).to_csv(args.output/'test_metrics.csv',index=False)
        completed.append(name)
        plot_results(args.output,classes,completed)
        del model;gc.collect()
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    print(f'Completed. Results and figures: {args.output.resolve()}',flush=True)

if __name__=='__main__':main()
