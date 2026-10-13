"""Fixed generated-video test inputs, rendered by the native H3 video decoder."""
import html
import json
from pathlib import Path
import subprocess
import time

import imageio.v3 as iio
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from h3dmd.decoders import load_h3_vaes
from h3dmd.stride_cache import atomic_json
from h3dmd.upsampling import frames, video, file_sha256


def visualization_sources(manifest):
    if not manifest:
        return []
    rows=json.loads(Path(manifest).read_text())
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Visualization requires unique fixed input IDs')
    for row in rows:
        if not row['id'].replace('_','').replace('-','').isalnum():
            raise ValueError('Unsafe visualization input ID')
        row['sha256']=file_sha256(row['path'])
    return rows


class UpsamplerVisualization:
    def __init__(self, settings, output, device, sources):
        self.settings,self.device,self.sources=settings,device,sources
        self.root=Path(output)/'visualization'
        self.reference=self.root/'reference'
        self.reference.mkdir(parents=True,exist_ok=True)
        self.vae=None

    @torch.inference_mode()
    def decode(self, latent, num_frames):
        if self.vae is None:
            self.vae=load_h3_vaes(self.settings['visualization_vae_path'],self.device,
                                  components=('video_vae',)).video_vae
        mean=torch.tensor(self.vae.config.latents_mean,device=self.device).view(1,24,1,1,1)
        std=torch.tensor(self.vae.config.latents_std,device=self.device).view(1,24,1,1,1)
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):
            decoded=self.vae.decode(latent.to(self.device).float()*std+mean,return_dict=False)[0]
        if not torch.isfinite(decoded).all():raise ValueError('Non-finite visualization video')
        pm=torch.tensor([.485,.456,.406],device=self.device).view(1,3,1,1,1)
        ps=torch.tensor([.229,.224,.225],device=self.device).view(1,3,1,1,1)
        pixels=((decoded.float()*ps+pm).clamp(0,1)[0].permute(1,2,3,0)*255).round().byte().cpu().numpy()
        if pixels.shape[0]!=num_frames:raise ValueError('Visualization decoded timeline changed')
        return pixels

    @staticmethod
    def write_video(path,pixels,fps):
        temporary=path.with_name(path.stem+'.tmp.mp4')
        iio.imwrite(temporary,pixels,fps=fps,codec='libx264',macro_block_size=1,ffmpeg_params=['-threads','2'])
        temporary.replace(path)

    @torch.inference_mode()
    def run(self,model,step):
        started=time.monotonic();model.eval()
        destination=self.root/f'iter_{step:09d}';destination.mkdir(parents=True,exist_ok=True)
        reports=[];sections=[]
        for row in self.sources:
            sample_started=time.monotonic();name=row['id'];path=Path(row['path'])
            if file_sha256(path)!=row['sha256']:raise ValueError('Visualization source changed')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            low=payload['video_latents'].float();meta=payload['metadata']
            if (low.ndim!=5 or tuple(low.shape[:2])!=(1,24) or tuple(low.shape[-2:])!=(20,32)
                    or not torch.isfinite(low).all() or (meta['width'],meta['height'])!=(512,320)):
                raise ValueError('Expected standardized generated H3 320p latent')
            n,fps=meta['num_frames'],meta.get('fps',24)
            if (n-5)%17 or low.shape[2]!=((n-5)//17)*5+2:
                raise ValueError('Visualization latent/frame alignment mismatch')
            reference_binding=self.reference/f'{name}.json'
            if reference_binding.exists() and json.loads(reference_binding.read_text())['sha256']!=row['sha256']:
                raise ValueError('Visualization reference binding changed')
            # Reference MP4s and exact decoded frame PNGs are cached across iterations.
            indices=[0,n//2,n-1]
            for tag in ['input320','bicubic480']:
                target=self.reference/f'{name}_{tag}.mp4'
                if not target.exists():
                    value=low if tag=='input320' else video(F.interpolate(frames(low),size=(30,48),mode='bicubic',align_corners=False),1,low.shape[2])
                    pixels=self.decode(value,n);self.write_video(target,pixels,fps)
                    for index in indices:Image.fromarray(pixels[index]).save(self.reference/f'{name}_{tag}_{index}.png')
                    del pixels
            atomic_json(reference_binding,dict(sha256=row['sha256'],source=str(path),metadata=meta))
            high=model(low.to(self.device))
            if tuple(high.shape)!=(1,24,low.shape[2],30,48):raise ValueError('Mapper visualization geometry changed')
            pixels=self.decode(high,n);target=destination/f'{name}_learned480.mp4'
            self.write_video(target,pixels,fps)
            sheet=Image.new('RGB',(1152,3*264),'#181818')
            for y,index in enumerate(indices):
                for x,tag in enumerate(['input320','bicubic480','learned480']):
                    image=Image.fromarray(pixels[index]) if tag=='learned480' else Image.open(self.reference/f'{name}_{tag}_{index}.png')
                    image=image.resize((384,240),Image.Resampling.BICUBIC)
                    sheet.paste(image,(x*384,y*264+24))
                    ImageDraw.Draw(sheet).text((x*384+6,y*264+5),f'{tag} frame {index}',fill='white')
            sheet.save(destination/f'{name}_contact.jpg')
            del pixels,high
            compare=destination/f'{name}_comparison.mp4'
            command=['ffmpeg','-nostdin','-y','-loglevel','error',
                     '-i',str(self.reference/f'{name}_input320.mp4'),
                     '-i',str(self.reference/f'{name}_bicubic480.mp4'),'-i',str(target),
                     '-filter_complex_threads','2','-filter_complex','[0:v]scale=768:480[a];[a][1:v][2:v]hstack=inputs=3[v]',
                     '-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','20','-threads','2',str(compare)]
            subprocess.run(command,check=True,timeout=300,stdout=subprocess.DEVNULL)
            report=dict(id=name,source=str(path),source_sha256=row['sha256'],num_frames=n,fps=fps,
                        learned=str(target),comparison=str(compare),contact=str(destination/f'{name}_contact.jpg'),
                        seconds=time.monotonic()-sample_started)
            reports.append(report)
            sections.append(f'<h2>{html.escape(name)}</h2><p>Left: 320p input (display resized); middle: bicubic latent 480p; right: Learned latent 480p.</p><video controls preload="metadata" width="1152" src="{name}_comparison.mp4"></video><br><img width="1152" src="{name}_contact.jpg">')
            print(json.dumps(dict(event='visualization_sample',step=step,**report)),flush=True)
        (destination/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>H3 upsampler</title><style>body{background:#151515;color:#eee;font-family:sans-serif}video,img{max-width:100%}</style>'+''.join(sections))
        report=dict(step=step,seconds=time.monotonic()-started,decoder='native_h3_decode_base_fp32_weights_fp16_autocast',
                    audio='not_modified_not_decoded',samples=reports)
        atomic_json(destination/'COMPLETE.json',report)
        steps=sorted(p.parent.name for p in self.root.glob('iter_*/COMPLETE.json'))
        links=''.join(f'<li><a href="{name}/index.html">{name}</a></li>' for name in reversed(steps))
        (self.root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>H3 upsampler previews</title>'
            '<style>body{background:#151515;color:#eee;font-family:sans-serif}a{color:#8ce0e9}</style>'
            '<h1>H3 upsampler · iteration previews</h1><ul>'+links+'</ul>')
        if self.settings.get('visualization_oss_prefix'):
            subprocess.run(['python3','-u','-m','scripts.publish_h3_upsampler_preview',
                '--root',str(self.root),'--step',str(step),
                '--prefix',self.settings['visualization_oss_prefix'],
                '--config',self.settings.get('visualization_oss_config','/mnt/data/butong/data_transfer/config.yaml')],
                check=True,timeout=900)
        return report
