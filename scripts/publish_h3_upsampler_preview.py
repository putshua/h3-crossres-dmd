"""Publish completed upsampler preview media; model/optimizer files stay local."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import oss2
import yaml
from scripts.publish_h3_dmd_gallery_oss import upload_one,make_public_read,direct_url


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--prefix',required=True)
    p.add_argument('--config',default='/mnt/data/butong/data_transfer/config.yaml')
    args=p.parse_args();root=Path(args.root)
    directory=root/f'iter_{args.step:09d}'
    if not (directory/'COMPLETE.json').is_file():raise ValueError('Preview is incomplete')
    config=yaml.safe_load(Path(args.config).read_text())
    bucket=oss2.Bucket(oss2.Auth(config['oss_upload_access_key_id'],config['oss_upload_access_key_secret']),
                      config['oss_upload_endpoint'],config['oss_upload_bucket'],session=oss2.Session(pool_size=4))
    types={'.mp4':'video/mp4','.jpg':'image/jpeg','.png':'image/png','.html':'text/html; charset=utf-8'}
    media=[f for f in directory.iterdir() if f.suffix in types and f.name!='index.html']
    def publish(file):
        key=args.prefix.rstrip('/')+'/'+file.relative_to(root).as_posix()
        upload_one(bucket,key,file,types[file.suffix]);make_public_read(bucket,key)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(publish,media))
    publish(directory/'index.html')
    # Include previous published iterations even when publishing from a new local copy.
    key=args.prefix.rstrip('/')+'/preview_manifest.json'
    try:previous=json.loads(bucket.get_object(key).read())
    except oss2.exceptions.NoSuchKey:previous={'steps':[]}
    steps=sorted(set(previous['steps'])|{args.step})
    manifest=dict(steps=steps,latest=max(steps))
    links=''.join(f'<li><a href="iter_{step:09d}/index.html">iter {step}</a></li>' for step in reversed(steps))
    index='<!doctype html><meta charset="utf-8"><title>H3 LTX upsampler previews</title><style>body{background:#151515;color:#eee;font-family:sans-serif}a{color:#8ce0e9}</style><h1>H3 LTX upsampler previews</h1><ul>'+links+'</ul>'
    for name,body,mime in [('preview_manifest.json',json.dumps(manifest),'application/json'),('index.html',index,'text/html; charset=utf-8')]:
        key=args.prefix.rstrip('/')+'/'+name
        bucket.put_object(key,body.encode(),headers={'Content-Type':mime,'Cache-Control':'no-cache'})
        make_public_read(bucket,key)
    url=direct_url(config['oss_upload_bucket'],args.prefix.rstrip('/')+'/index.html')
    (root/'preview_url.txt').write_text(url+'\n');print('PREVIEW_URL='+url,flush=True)


if __name__=='__main__':main()
