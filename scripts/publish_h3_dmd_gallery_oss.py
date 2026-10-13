#!/usr/bin/env python3
"""Publish the four-resolution, 32-prompt H3 DMD evaluation as an OSS gallery."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from urllib.parse import quote

import oss2
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/h3-dmd-mixed331-lr1e5-f2e6-20260920"
DEFAULT_CONFIG = Path("/mnt/data/butong/data_transfer/config.yaml")
DEFAULT_PREFIX = "butong/h3_dmd_mixed331_iter700_4res_gallery_20260924"
PUBLIC_ENDPOINT = "https://oss-cn-wulanchabu.aliyuncs.com"
RESOLUTIONS = {
    "512x320": "eval_iter700_local4_20260923",
    "512x384": "eval_iter700_512x384_local4_20260923",
    "832x480": "eval_iter700_832x480_local4_20260923",
    "1344x768": "eval_iter700_1344x768_local4_20260923",
}
IDS = [f"high_motion_{i:02d}" for i in range(1, 33)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--oss-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--expires-days", type=int, default=300)
    parser.add_argument("--public-read", action="store_true", help="Use anonymous OSS direct links and public-read object ACLs")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def index_inputs(run_dir: Path) -> list[dict]:
    entries = []
    for prompt_id in IDS:
        videos = {}
        prompt = None
        prompt_hash = None
        for resolution, dirname in RESOLUTIONS.items():
            stem = f"{prompt_id}_{resolution}_243f_seed1"
            directory = run_dir / dirname
            video = directory / f"{stem}.mp4"
            sidecar = directory / f"{stem}.json"
            if not video.is_file() or not sidecar.is_file():
                raise FileNotFoundError(f"Missing evaluation output: {video} or {sidecar}")
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta["num_forwards"] != 4 or meta["solver"] != "renoise":
                raise ValueError(f"Not the expected four-step DMD output: {sidecar}")
            if (meta["width"], meta["height"]) != tuple(map(int, resolution.split("x"))):
                raise ValueError(f"Resolution mismatch in {sidecar}")
            if prompt is None:
                prompt, prompt_hash = meta["prompt"], meta["prompt_sha256"]
            elif (meta["prompt"], meta["prompt_sha256"]) != (prompt, prompt_hash):
                raise ValueError(f"Prompt mismatch across resolutions: {prompt_id}")
            videos[resolution] = video
        entries.append({"id": prompt_id, "prompt": prompt, "videos": videos})
    return entries


def prepare_posters(entries: list[dict], gallery_dir: Path) -> None:
    poster_dir = gallery_dir / "posters"
    poster_dir.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(entries, 1):
        poster = poster_dir / f"{entry['id']}.jpg"
        if not poster.is_file() or poster.stat().st_size == 0:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "5",
                 "-i", str(entry["videos"]["1344x768"]), "-frames:v", "1",
                 "-vf", "scale=560:-2", "-q:v", "3", str(poster)],
                check=True,
            )
        entry["poster"] = poster
        if index % 8 == 0:
            print(f"Prepared posters {index}/32", flush=True)


def gallery_html(entries: list[dict], access_note: str) -> str:
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__GALLERY_DATA__", payload).replace("__ACCESS_NOTE__", access_note)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 DMD · 4 resolutions × 32 prompts</title>
<style>
:root{color-scheme:dark;--bg:#0b1018;--panel:#141d29;--line:#2b394b;--text:#edf4fb;--muted:#a3b4c6;--accent:#70d4e8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;padding:18px max(20px,calc((100vw - 1600px)/2));background:#0b1018ed;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}
h1{font-size:24px;letter-spacing:.01em;margin:0 0 4px}.lead{color:var(--muted);margin:0 0 15px}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input,select,button{font:inherit;color:var(--text);background:#1c2938;border:1px solid #41556b;border-radius:8px;padding:9px 11px}input{min-width:240px;flex:1}button{cursor:pointer}button:hover{background:#29435b}label{color:var(--muted)}
main{max-width:1640px;margin:auto;padding:22px 20px 50px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:13px;overflow:hidden}
.card-head{display:flex;align-items:center;gap:10px;padding:12px 14px}.id{font-weight:700;color:var(--accent);font-variant-numeric:tabular-nums}.card-head select{margin-left:auto;padding:5px 8px}.card video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#030609}.card-body{padding:11px 14px 14px}.prompt{color:#dce7f4;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;overflow:hidden;white-space:pre-wrap;min-height:42px}.actions{display:flex;gap:8px;align-items:center;margin-top:12px}.hint{margin-left:auto;color:var(--muted);font-size:12px}
.empty{padding:70px;text-align:center;color:var(--muted)}dialog{width:min(1500px,96vw);max-height:94vh;padding:0;background:#111b27;color:var(--text);border:1px solid #536b82;border-radius:13px;box-shadow:0 25px 90px #000b}dialog::backdrop{background:#02060bc9}
.modal-head{display:flex;align-items:center;gap:10px;padding:12px 16px;border-bottom:1px solid var(--line)}.modal-head strong{color:var(--accent)}.close{margin-left:auto}.modal-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px 16px}.compare-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;padding:0 16px 16px}.compare-grid video{width:100%;max-height:60vh;background:#030609;border-radius:8px}.compare-grid label{display:block;margin:0 0 5px}.modal-prompt{padding:0 16px 15px;color:var(--muted);white-space:pre-wrap;max-height:100px;overflow:auto}
@media(max-width:900px){.grid,.compare-grid{grid-template-columns:1fr}header{padding:14px 16px}main{padding:16px}.compare-grid video{max-height:35vh}}
</style></head><body>
<header><h1>H3 DMD · 四种分辨率</h1><p class="lead">Iter 700 · 同一批 32 条 high-motion prompts · 4-step renoise · seed 1 · 243 帧 / 24 fps · 点击视频直接播放</p>
<div class="toolbar"><input id="search" type="search" placeholder="搜索编号或 prompt 内容"><label for="global-res">统一分辨率</label><select id="global-res"><option>512x320</option><option>512x384</option><option selected>832x480</option><option>1344x768</option></select><span id="count"></span></div>
<div class="lead" style="margin:8px 0 0;font-size:12px">__ACCESS_NOTE__ · 并排对比时右侧默认静音，可在播放器中开启</div></header>
<main><div class="grid" id="grid"></div></main>
<dialog id="compare"><div class="modal-head"><strong id="compare-id"></strong><span>分辨率并排对比</span><button class="close" id="close">关闭</button></div>
<div class="modal-controls"><label>左侧 <select id="left-res"><option>512x320</option><option>512x384</option><option>832x480</option><option>1344x768</option></select></label><label>右侧 <select id="right-res"><option>512x320</option><option>512x384</option><option>832x480</option><option>1344x768</option></select></label><button id="sync-play">同步播放</button><button id="sync-pause">同步暂停</button><button id="sync-restart">从头播放</button></div>
<div class="compare-grid"><div><label id="left-label"></label><video id="left-video" controls playsinline preload="none"></video></div><div><label id="right-label"></label><video id="right-video" controls playsinline preload="none" muted></video></div></div><div class="modal-prompt" id="compare-prompt"></div></dialog>
<script>
const DATA=__GALLERY_DATA__, RES=['512x320','512x384','832x480','1344x768'];
const grid=document.getElementById('grid'), search=document.getElementById('search'), globalRes=document.getElementById('global-res');
const modal=document.getElementById('compare'), left=document.getElementById('left-video'), right=document.getElementById('right-video');
let current=null;
function optionSelect(value){const s=document.createElement('select');for(const r of RES){const o=document.createElement('option');o.value=o.textContent=r;s.append(o)}s.value=value;return s}
function loadCardVideo(card,entry,res){const video=card.querySelector('video');video.pause();video.poster=entry.poster;video.src=entry.videos[res];video.load()}
function render(){const q=search.value.trim().toLowerCase(),matches=DATA.filter(x=>x.id.includes(q)||x.prompt.toLowerCase().includes(q));grid.replaceChildren();document.getElementById('count').textContent=`${matches.length} / ${DATA.length} 条`;
if(!matches.length){const e=document.createElement('div');e.className='empty';e.textContent='没有匹配的 prompt';grid.append(e);return}
const obs=new IntersectionObserver(items=>{for(const item of items){if(item.isIntersecting){const card=item.target;if(!card.dataset.loaded){loadCardVideo(card,card.entry,card.res.value);card.dataset.loaded='1'}obs.unobserve(card)}}},{rootMargin:'550px'});
for(const entry of matches){const card=document.createElement('article');card.className='card';card.entry=entry;const head=document.createElement('div');head.className='card-head';const id=document.createElement('span');id.className='id';id.textContent=entry.id;const sel=optionSelect(globalRes.value);card.res=sel;head.append(id,sel);const video=document.createElement('video');video.controls=true;video.playsInline=true;video.preload='none';video.poster=entry.poster;const body=document.createElement('div');body.className='card-body';const prompt=document.createElement('div');prompt.className='prompt';prompt.textContent=entry.prompt;prompt.title=entry.prompt;const actions=document.createElement('div');actions.className='actions';const compare=document.createElement('button');compare.textContent='并排对比';compare.onclick=()=>openCompare(entry,sel.value);const hint=document.createElement('span');hint.className='hint';hint.textContent='音频可在播放器中控制';actions.append(compare,hint);body.append(prompt,actions);card.append(head,video,body);sel.onchange=()=>{if(card.dataset.loaded)loadCardVideo(card,entry,sel.value)};grid.append(card);obs.observe(card)}
}
function setModalVideo(video,res){video.pause();video.src=current.videos[res];video.poster=current.poster;video.load()}
function openCompare(entry,res){current=entry;document.getElementById('compare-id').textContent=entry.id;document.getElementById('compare-prompt').textContent=entry.prompt;document.getElementById('left-res').value=res;document.getElementById('right-res').value=res==='1344x768'?'832x480':'1344x768';refreshModal();modal.showModal()}
function refreshModal(){if(!current)return;const l=document.getElementById('left-res').value,r=document.getElementById('right-res').value;document.getElementById('left-label').textContent=l;document.getElementById('right-label').textContent=r;setModalVideo(left,l);setModalVideo(right,r)}
document.getElementById('left-res').onchange=refreshModal;document.getElementById('right-res').onchange=refreshModal;
document.getElementById('sync-play').onclick=()=>{right.currentTime=left.currentTime;left.play().catch(()=>{});right.play().catch(()=>{})};document.getElementById('sync-pause').onclick=()=>{left.pause();right.pause()};document.getElementById('sync-restart').onclick=()=>{left.currentTime=0;right.currentTime=0;left.play().catch(()=>{});right.play().catch(()=>{})};
document.getElementById('close').onclick=()=>modal.close();modal.onclose=()=>{left.pause();right.pause();left.removeAttribute('src');right.removeAttribute('src');left.load();right.load();current=null};
search.oninput=render;globalRes.onchange=render;render();
</script></body></html>'''


def upload_one(bucket: oss2.Bucket, key: str, path: Path, content_type: str) -> bool:
    size = path.stat().st_size
    try:
        meta = bucket.get_object_meta(key)
        if int(meta.headers.get("Content-Length", -1)) == size:
            return False
    except oss2.exceptions.NoSuchKey:
        pass
    bucket.put_object_from_file(
        key, str(path), headers={"Content-Type": content_type, "Content-Disposition": "inline"}
    )
    meta = bucket.get_object_meta(key)
    if int(meta.headers.get("Content-Length", -1)) != size:
        raise RuntimeError(f"Upload size mismatch: {key}")
    return True


def direct_url(bucket_name: str, key: str) -> str:
    return f"https://{bucket_name}.{PUBLIC_ENDPOINT.removeprefix('https://')}/{quote(key, safe='/')}"


def make_public_read(bucket: oss2.Bucket, key: str) -> None:
    bucket.put_object_acl(key, oss2.OBJECT_ACL_PUBLIC_READ)
    if bucket.get_object_acl(key).acl != oss2.OBJECT_ACL_PUBLIC_READ:
        raise RuntimeError(f"Public-read ACL verification failed: {key}")


def main() -> int:
    args = parse_args()
    if args.expires_days <= 0:
        raise ValueError("expires-days must be positive")
    run_dir = args.run_dir.resolve()
    gallery_dir = run_dir / "gallery_4res_iter700_20260924"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    entries = index_inputs(run_dir)
    prepare_posters(entries, gallery_dir)
    prefix = args.oss_prefix.strip("/")
    if not prefix:
        raise ValueError("OSS prefix cannot be empty")
    if args.prepare_only:
        local_entries = [
            {"id": e["id"], "prompt": e["prompt"],
             "poster": e["poster"].relative_to(gallery_dir).as_posix(),
             "videos": {r: "../" + p.relative_to(run_dir).as_posix() for r, p in e["videos"].items()}}
            for e in entries
        ]
        path = gallery_dir / "index.local.html"
        path.write_text(gallery_html(local_entries, "本地预览"), encoding="utf-8")
        print(f"LOCAL_PREVIEW={path}", flush=True)
        return 0

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    auth = oss2.Auth(config["oss_upload_access_key_id"], config["oss_upload_access_key_secret"])
    upload = oss2.Bucket(auth, config["oss_upload_endpoint"], config["oss_upload_bucket"], session=oss2.Session(pool_size=4))
    public = oss2.Bucket(auth, PUBLIC_ENDPOINT, config["oss_upload_bucket"])
    expiry = args.expires_days * 24 * 60 * 60
    expires_at = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=expiry)).isoformat()
    def object_url(key: str) -> str:
        return direct_url(config["oss_upload_bucket"], key) if args.public_read else public.sign_url("GET", key, expiry, slash_safe=True)
    uploaded = reused = 0
    manifest = []
    for index, entry in enumerate(entries, 1):
        item = {"id": entry["id"], "prompt": entry["prompt"], "videos": {}}
        poster_key = f"{prefix}/posters/{entry['id']}.jpg"
        changed = upload_one(upload, poster_key, entry["poster"], "image/jpeg")
        if args.public_read:
            make_public_read(upload, poster_key)
        uploaded += changed
        reused += not changed
        item["poster"] = object_url(poster_key)
        for resolution, path in entry["videos"].items():
            key = f"{prefix}/videos/{resolution}/{path.name}"
            changed = upload_one(upload, key, path, "video/mp4")
            if args.public_read:
                make_public_read(upload, key)
            uploaded += changed
            reused += not changed
            item["videos"][resolution] = object_url(key)
        manifest.append(item)
        if index % 4 == 0:
            print(f"Verified {index}/32 prompts; uploaded={uploaded}, reused={reused}", flush=True)

    access_note = "无签名 OSS 直链" if args.public_read else f"视频地址有效期至 {expires_at}"
    document = gallery_html(manifest, access_note).encode("utf-8")
    local_index = gallery_dir / "index.html"
    local_index.write_bytes(document)
    (gallery_dir / "gallery_manifest.json").write_text(
        json.dumps({"count": len(manifest), "access": "public-read" if args.public_read else "signed",
                    "expires_at": None if args.public_read else expires_at, "items": manifest},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    index_key = f"{prefix}/index.html"
    upload.put_object(index_key, document, headers={"Content-Type": "text/html; charset=utf-8", "Content-Disposition": "inline", "Cache-Control": "no-store"})
    if args.public_read:
        make_public_read(upload, index_key)
    if int(upload.get_object_meta(index_key).headers.get("Content-Length", -1)) != len(document):
        raise RuntimeError("Gallery HTML upload verification failed")
    url = object_url(index_key)
    (gallery_dir / "preview_url.txt").write_text(url + "\n", encoding="utf-8")
    print(f"OSS_PREFIX=oss://{config['oss_upload_bucket']}/{prefix}/")
    print(f"INDEX={local_index}")
    print(f"URL_FILE={gallery_dir / 'preview_url.txt'}")
    print(f"GALLERY_URL={url}")
    if not args.public_read:
        print(f"EXPIRES_AT={expires_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
