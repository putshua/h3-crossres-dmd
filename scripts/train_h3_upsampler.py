"""Build immutable committed-cache indexes or train H3 spatial latent upsamplers."""
import argparse
import datetime
import os
from pathlib import Path
import torch
import torch.distributed as dist


def main():
    parser=argparse.ArgumentParser(__doc__)
    subs=parser.add_subparsers(dest='action',required=True)
    p=subs.add_parser('snapshot')
    p.add_argument('--cache-root',required=True);p.add_argument('--output',required=True)
    p=subs.add_parser('train')
    p.add_argument('--config',default='configs/h3_upsampler_train_320to480.json')
    p.add_argument('--index',required=True);p.add_argument('--output',required=True)
    p.add_argument('--set',nargs='*',default=[])
    args=parser.parse_args();torch.set_num_threads(2)
    if args.action=='snapshot':
        from h3dmd.upsampler_data import snapshot_cache
        result=snapshot_cache(args.cache_root,args.output)
        print({k:v for k,v in result.items() if k!='journal_boundaries'})
        return
    from h3dmd.upsampler_training import load_settings,train
    settings,config=load_settings(args.config,args.set)
    local,rank,world=(int(os.environ.get(k,default)) for k,default in [('LOCAL_RANK','0'),('RANK','0'),('WORLD_SIZE','1')])
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group(backend='nccl' if device.type=='cuda' else 'gloo', timeout=datetime.timedelta(minutes=30))
    try:train(settings,config,Path(args.index),Path(args.output),device,rank,world)
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
