"""Render fixed H3 test outputs from a saved upsampler checkpoint on a free GPU."""
import argparse
import json
from pathlib import Path
import torch

from h3dmd.mapper_pretraining import atomic_save
from h3dmd.upsampler_training import load_settings
from h3dmd.upsampling import new_mapper,mapper_spec
from h3dmd.upsampler_visualization import UpsamplerVisualization,visualization_sources


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--manifest',help='Override fixed preview inputs, including previews disabled during a pilot')
    p.add_argument('--oss-prefix',default='')
    args=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(args.gpu);device=torch.device('cuda',args.gpu)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    settings=dict(saved['signature']['settings'])
    _,config=load_settings(settings)
    if mapper_spec(config)!=saved['signature']['spec']:raise ValueError('Checkpoint mapper specification mismatch')
    module=new_mapper(config).to(device);module.load_state_dict(saved['model'])
    output=Path(args.output).resolve()
    destination=output/'visualization'/f"iter_{saved['step']:09d}"
    if (destination/'COMPLETE.json').exists():raise FileExistsError('Visualization already completed')
    # Keep a small exact model snapshot so this manual evaluation is reproducible.
    atomic_save(dict(step=saved['step'],spec=saved['signature']['spec'],model=saved['model']),destination/'mapper.pt')
    manifest=args.manifest or settings.get('visualization_manifest')
    sources=visualization_sources(manifest) if manifest else saved['signature']['visualization_sources']
    if not sources:raise ValueError('No preview inputs; supply --manifest')
    if args.oss_prefix:settings['visualization_oss_prefix']=args.oss_prefix
    callback=UpsamplerVisualization(settings,output,device,sources)
    callback.run(module,saved['step'])

if __name__=='__main__':main()
