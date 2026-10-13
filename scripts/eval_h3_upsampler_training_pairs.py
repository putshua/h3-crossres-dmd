"""Render full training pairs and native 480p GT from an immutable model snapshot."""
import argparse
import html
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from h3dmd.mapper_pretraining import atomic_save
from h3dmd.stride_cache import atomic_json
from h3dmd.upsampler_data import IndexedPairs, ResumeBatches, open_index
from h3dmd.upsampler_training import load_settings
from h3dmd.upsampler_visualization import UpsamplerVisualization
from h3dmd.upsampling import new_mapper, mapper_spec, file_sha256, reconstruction_losses, frames, video


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda', args.gpu)
    if args.output.exists():
        raise FileExistsError('Use a new evaluation directory')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    settings = saved['signature']['settings']
    _, config = load_settings('configs/h3_upsampler_train_320to480.json',
                             [f'{k}={json.dumps(v)}' for k, v in settings.items()])
    if mapper_spec(config) != saved['signature']['spec']:
        raise ValueError('Mapper specification mismatch')
    if file_sha256(args.index) != saved['signature']['index_sha256']:
        raise ValueError('Training index changed')
    args.output.mkdir(parents=True)
    atomic_save(saved, args.output/'checkpoint.pt')
    checkpoint_sha = file_sha256(args.output/'checkpoint.pt')
    model = new_mapper(config).to(device).eval()
    model.load_state_dict(saved['model'])
    dataset = IndexedPairs(args.index, config, 'train')
    # Pick one clip per duration from the actual already-consumed rank-0 stream.
    selected = {}
    with open_index(args.index) as db:
        batches = ResumeBatches(len(dataset), settings['batch_size'], 0, saved['step'],
                                config.training.seed, 0, saved['signature']['world_size'])
        for training_step, batch in enumerate(batches, 1):
            for dataset_index, crop_seed in batch:
                ordinal = int(dataset.ordinals[dataset_index])
                row = db.execute('SELECT id,seconds,source_group,manifest,offset,length FROM pairs WHERE ordinal=?',
                                 (ordinal,)).fetchone()
                if row[1] not in selected:
                    selected[row[1]] = (dataset_index, row, training_step, crop_seed)
            if len(selected) == 3:
                break
    if set(selected) != {5, 10, 15}:
        raise ValueError('Completed training stream lacks all three durations')
    decoder = UpsamplerVisualization(settings, args.output, device, [])
    reports, sections = [], []
    for seconds, (index, record, seen_step, crop_seed) in sorted(selected.items()):
        sample_id, _, group, manifest, offset, length = record
        with open(manifest, 'rb') as handle:
            handle.seek(offset)
            source = json.loads(handle.read(length))
        low, gt = dataset[index]  # Full cached clip, without training crop/flip.
        low, gt = low[None].to(device), gt[None].to(device)
        n = 17*((low.shape[2]-2)//5)+5
        prediction = model(low)
        bicubic = video(F.interpolate(frames(low), size=(30,48), mode='bicubic',
                                      align_corners=False), 1, low.shape[2])
        if prediction.shape != gt.shape or not torch.isfinite(prediction).all():
            raise ValueError('Invalid mapper prediction')
        learned_loss, learned_parts = reconstruction_losses(prediction, gt)
        bicubic_loss, bicubic_parts = reconstruction_losses(bicubic, gt)
        name = f'train_{seconds}s'
        indices = [0, n//2, n-1]
        sheet = Image.new('RGB', (1536, 3*264), '#181818')
        tags = ['input320', 'bicubic480', 'learned480', 'gt480']
        for col, (tag, latent) in enumerate(zip(tags, [low, bicubic, prediction, gt])):
            pixels = decoder.decode(latent, n)
            decoder.write_video(args.output/f'{name}_{tag}.mp4', pixels, 24)
            for y, frame_index in enumerate(indices):
                frame = Image.fromarray(pixels[frame_index])
                frame.save(args.output/f'{name}_{tag}_frame{frame_index}.png')
                sheet.paste(frame.resize((384,240), Image.Resampling.BICUBIC), (col*384,y*264+24))
                ImageDraw.Draw(sheet).text((col*384+6,y*264+5), f'{tag} frame {frame_index}', fill='white')
            del pixels
        sheet.save(args.output/f'{name}_contact.jpg')
        command = ['ffmpeg','-nostdin','-y','-loglevel','error']
        for tag in tags:
            command += ['-i', str(args.output/f'{name}_{tag}.mp4')]
        command += ['-filter_complex_threads','2','-filter_complex',
                    '[0:v]scale=768:480[a];[a][1:v][2:v][3:v]hstack=inputs=4[v]',
                    '-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','20',
                    '-threads','2',str(args.output/f'{name}_comparison.mp4')]
        subprocess.run(command, check=True, timeout=300)
        for tag in tags + ['comparison']:
            probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,nb_frames','-of','json',
                str(args.output/f'{name}_{tag}.mp4')],text=True))['streams'][0]
            expected = (512,320) if tag == 'input320' else ((3072,480) if tag == 'comparison' else (768,480))
            if (probe['width'],probe['height']) != expected or int(probe['nb_frames']) != n:
                raise ValueError('Exported video geometry/timeline mismatch')
        report = dict(id=sample_id, seconds=seconds, num_frames=n, source_group=group,
                      split='train', original_rank0_training_step=seen_step,
                      original_training_crop_seed=crop_seed, evaluation='full cached clip; no augmentation',
                      source=source, learned_loss=float(learned_loss), bicubic_loss=float(bicubic_loss),
                      learned={k:float(v) for k,v in learned_parts.items()},
                      bicubic={k:float(v) for k,v in bicubic_parts.items()},
                      mse_ratio=float(learned_parts['mse']/bicubic_parts['mse']))
        reports.append(report)
        atomic_json(args.output/f'{name}_report.json',report)
        sections.append(f'<h2>{seconds}s · {html.escape(sample_id)}</h2>'
                        f'<p>Input 320p | Bicubic latent 480p | Learned latent 480p | Real-video VAE GT 480p</p>'
                        f'<video controls preload="metadata" width="1536" src="{name}_comparison.mp4"></video>'
                        f'<br><img width="1536" src="{name}_contact.jpg">')
        print(json.dumps(dict(seconds=seconds,mse_ratio=report['mse_ratio'],status='complete')),flush=True)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>H3 training pairs</title>'
        '<style>body{background:#151515;color:#eee;font-family:sans-serif}video,img{max-width:100%}</style>'
        f'<h1>Training pairs · checkpoint {saved["step"]}</h1><p>Silent video-only evaluation.</p>'+''.join(sections))
    atomic_json(args.output/'COMPLETE.json',dict(step=saved['step'],checkpoint_sha256=checkpoint_sha,
                source_checkpoint=str(args.checkpoint),columns=tags,samples=reports,media_validation_passed=True))


if __name__ == '__main__':
    main()
