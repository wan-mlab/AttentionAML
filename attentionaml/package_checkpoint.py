"""Package trusted server state_dict weights with explicit feature/class order."""
import argparse,json
from collections.abc import Mapping
from pathlib import Path
import torch
from .training import build_model,save_checkpoint,load_checkpoint
from .data import sha256


def ordered_list(path,key):
    obj=json.loads(Path(path).read_text())
    values=obj[key] if isinstance(obj,dict) else obj
    if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v.strip() for v in values):
        raise ValueError(f'{key} must be a nonempty JSON list of strings, in training order.')
    if len(values)!=len(set(values)):raise ValueError(f'Duplicate {key}.')
    return values


def package(weights,config_path,genes_path,classes_path,metadata_path,output):
    output=Path(output)
    if output.exists():raise ValueError('Output already exists.')
    cfg=json.loads(Path(config_path).read_text())
    if cfg['name'] not in ('MLP','AttentionAML'):raise ValueError('Only neural state dictionaries are supported.')
    genes=ordered_list(genes_path,'genes');classes=ordered_list(classes_path,'classes')
    if len(classes)<2:raise ValueError('At least two classes required.')
    metadata=json.loads(Path(metadata_path).read_text())
    if not isinstance(metadata,dict):raise ValueError('Metadata must be a JSON object.')
    for key in ['training_scope','input_transform','provenance']:
        if not isinstance(metadata.get(key),str) or not metadata[key].strip():raise ValueError(f'Metadata requires {key}.')
    if metadata['input_transform']!='log2(TPM+1)':raise ValueError('This inference pipeline expects log2(TPM+1).')
    raw=torch.load(weights,map_location='cpu',weights_only=True)
    if not isinstance(raw,Mapping):raise ValueError('Expected a state dictionary or dictionary containing state_dict/model_state_dict.')
    for key,expected in [('genes',genes),('classes',classes)]:
        if key in raw and raw[key]!=expected:raise ValueError(f'Embedded {key} differ from supplied order.')
    state=raw.get('state_dict',raw.get('model_state_dict',raw))
    if not isinstance(state,Mapping) or not state or not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ValueError('Unsupported weights; export model.state_dict() from the trusted training environment.')
    if all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
    if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Weights contain nonfinite values.')
    model=build_model(cfg,len(genes),len(classes));model.load_state_dict(state,strict=True)
    metadata.update(source_weights_sha256=sha256(weights),source_config_sha256=sha256(config_path),
                    feature_order_sha256=sha256(genes_path),class_order_sha256=sha256(classes_path))
    output.parent.mkdir(parents=True,exist_ok=True)
    save_checkpoint(output,model,cfg,genes,classes,metadata)
    restored,payload=load_checkpoint(output)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    manifest={'file':output.name,'sha256':sha256(output),'model':cfg['name'],'n_genes':len(genes),'n_classes':len(classes),'metadata':metadata}
    output.with_suffix(output.suffix+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Packaged {cfg["name"]}: {len(genes)} genes, {len(classes)} classes -> {output}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['weights','config','genes','classes','metadata','output']:p.add_argument('--'+flag,required=True,type=Path)
    a=p.parse_args();package(a.weights,a.config,a.genes,a.classes,a.metadata,a.output)
if __name__=='__main__':main()
