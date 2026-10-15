# /// script
# requires-python = ">=3.10"
# dependencies = ["rembg[cpu]>=2.0.60", "pillow"]
# ///
"""
Local image generation with FLUX 3 Image Klein (distilled) through a headless ComfyUI.

    uv run --script flux3.py "Studio photograph of a ..." -o out/teapot -n 4
    python flux3.py -f prompt.txt -o out/hero --size 16:9
    python flux3.py "make the cup red" --ref out/teapot/teapot_42.png -o out/edit
    python flux3.py --jobs jobs.json            # many assets in one server session
    python flux3.py --stop                      # stop a server left running with --keep

If a ComfyUI server already listens on FLUX3_PORT (default 8188) it is used as is. Otherwise
the ComfyUI Windows portable build at FLUX3_COMFY_DIR is started on demand and stopped after
the batch (unless --keep). Every PNG gets a sidecar .json with prompt, seed and settings, so a
good result can be reproduced or refined later.

Prompts: one prose caption, medium and style first, frame-relative layout words instead of
coordinates (the model never sees bounding boxes; BFL's prompt upsampler turns them into
words too). See skills/imagegen/SKILL.md for the full prompt guide.
"""

import argparse
import ctypes
import json
import mimetypes
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY = Path(os.environ.get("FLUX3_COMFY_DIR") or ROOT / "ComfyUI_windows_portable")  # empty = unset
PORT = int(os.environ.get("FLUX3_PORT", "8188"))
URL = f"http://127.0.0.1:{PORT}"
# Inside a Claude Code plugin the root is replaced on every update, so state goes to the plugin data dir.
LOGS = Path(os.environ.get("CLAUDE_PLUGIN_DATA") or ROOT) / "logs"
PID_FILE = LOGS / "comfyui.pid"

UNET = "flux3_image_klein_sd_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_4b_int8_convrot.safetensors"
VAE = "flux2-vae.safetensors"
RATIOS = {"1:1": (1024, 1024), "4:3": (1184, 880), "3:4": (880, 1184), "3:2": (1248, 832),
          "2:3": (832, 1248), "16:9": (1360, 768), "9:16": (768, 1360)}


# ---------- server ----------

def _get(path, timeout=10):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return r.read()


def server_up():
    try:
        _get("/system_stats", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def start_server(extra_args=()):
    """Start ComfyUI in the background; returns the Popen handle."""
    LOGS.mkdir(exist_ok=True)
    python = COMFY / "python_embeded" / "python.exe"
    if not python.exists():
        sys.exit(f"No ComfyUI server on port {PORT} and no portable build at {COMFY}. "
                 "Start ComfyUI yourself or set FLUX3_COMFY_DIR to the ComfyUI_windows_portable folder.")
    args = [str(python), "-s", str(COMFY / "ComfyUI" / "main.py"), "--listen", "127.0.0.1",
            "--port", str(PORT), "--disable-auto-launch", *extra_args,
            *os.environ.get("FLUX3_COMFY_ARGS", "").split()]
    log = open(LOGS / "comfyui.log", "ab")
    proc = subprocess.Popen(args, cwd=COMFY, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    PID_FILE.write_text(str(proc.pid))
    print(f"starting ComfyUI (pid {proc.pid}, log {LOGS / 'comfyui.log'}) ...", flush=True)
    t0 = time.time()
    while not server_up():
        if proc.poll() is not None:
            sys.exit(f"ComfyUI exited with code {proc.returncode}; see {LOGS / 'comfyui.log'}")
        if time.time() - t0 > 180:
            proc.kill()
            sys.exit(f"ComfyUI did not come up in 180 s; see {LOGS / 'comfyui.log'}")
        time.sleep(1)
    print(f"ComfyUI ready in {time.time() - t0:.0f} s", flush=True)
    return proc


def stop_server(proc=None):
    pid = proc.pid if proc else (int(PID_FILE.read_text()) if PID_FILE.exists() else None)
    if pid is None:
        print("no server pid recorded")
        return
    # taskkill /T also ends child processes; the server holds ~11 GB that must be freed.
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    PID_FILE.unlink(missing_ok=True)
    print(f"ComfyUI stopped (pid {pid})")


# ---------- resource sampling (--stats) ----------

class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_ram_gb():
    if os.name != "nt":
        return float("nan")
    s = _MemStatus()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 2**30


class Sampler(threading.Thread):
    """Polls VRAM (nvidia-smi) and free RAM to report peaks for a run."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_vram_mb, self.min_free_ram_gb, self._halt = 0, 1e9, threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.peak_vram_mb = max(self.peak_vram_mb, int(out.split()[0]))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            self.min_free_ram_gb = min(self.min_free_ram_gb, free_ram_gb())
            self._halt.wait(0.5)

    def stop(self):
        self._halt.set()
        self.join(2)
        return f"peak VRAM {self.peak_vram_mb / 1024:.1f} GB, min free RAM {self.min_free_ram_gb:.1f} GB"


# ---------- graph ----------

def parse_size(size):
    if size in RATIOS:
        return RATIOS[size]
    w, _, h = size.lower().partition("x")
    w, h = int(w), int(h)
    return w - w % 16, h - h % 16


def build_graph(prompt, seed, steps, size, ref_names, prefix):
    """API-format graph mirroring BFL's image_flux3_klein_distilled_{t2i,image_edit} templates."""
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "flux3", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }
    sampler = {"model": ["unet", 0], "seed": seed, "steps": steps, "cfg": 1.0,
               "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}
    if ref_names:
        g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
        refs = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
                "resolution": "resize", "resolution.megapixels": 1.0}
        for i, name in enumerate(ref_names):
            g[f"load{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            refs[f"images.image{i}"] = [f"load{i}", 0]
        g["refs"] = {"class_type": "ReferenceMultiLatent", "inputs": refs}
        sampler.update(positive=["refs", 0], negative=["refs", 1])
        # Without --size the output takes the first reference's (1 MP) shape, as in BFL's template.
        sampler["latent_image"] = ["latent", 0] if size else ["refs", 2]
    else:
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": ""}}
        sampler.update(positive=["pos", 0], negative=["neg", 0], latent_image=["latent", 0])
        size = size or (1024, 1024)
    if size:
        g["latent"] = {"class_type": "EmptyFlux2LatentImage",
                       "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
    g["sampler"] = {"class_type": "KSampler", "inputs": sampler}
    return g


# ---------- API calls ----------

def upload(path):
    path = Path(path)
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: {mimetypes.guess_type(path.name)[0] or 'image/png'}\r\n\r\n".encode(),
        path.read_bytes(), f"\r\n--{boundary}--\r\n".encode()])
    req = urllib.request.Request(URL + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        info = json.loads(r.read())
    return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]


def submit(graph):
    req = urllib.request.Request(URL + "/prompt", data=json.dumps({"prompt": graph, "client_id": "flux3"}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        sys.exit(f"ComfyUI rejected the graph:\n{e.read().decode(errors='replace')}")


def wait(prompt_id, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = json.loads(_get(f"/history/{prompt_id}"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                sys.exit(f"generation failed: {json.dumps(msgs, indent=1)[:3000]}")
            return entry, time.time() - t0
        time.sleep(1)
    sys.exit(f"timed out after {timeout} s waiting for {prompt_id}")


def fetch(image, dest):
    q = urllib.parse.urlencode({"filename": image["filename"], "subfolder": image["subfolder"], "type": image["type"]})
    dest.write_bytes(_get(f"/view?{q}", timeout=60))


# ---------- post-processing ----------

_rembg_session = None


def remove_background(src, dest):
    global _rembg_session
    from rembg import new_session, remove  # imported lazily: only --nobg needs it
    from PIL import Image
    if _rembg_session is None:
        _rembg_session = new_session(os.environ.get("FLUX3_REMBG_MODEL", "isnet-general-use"))
    cut = remove(Image.open(src), session=_rembg_session)
    # Trim transparent margins; rembg leaves faint alpha (1-5) in empty areas, so threshold it.
    bbox = cut.getchannel("A").point(lambda a: 255 if a > 16 else 0).getbbox()
    (cut.crop(bbox) if bbox else cut).save(dest)


# ---------- jobs ----------

def run_job(job, defaults):
    """job: {name, prompt, out, n, seed, size, steps, refs, nobg}; returns list of written PNGs."""
    j = {**defaults, **{k: v for k, v in job.items() if v is not None}}
    out = Path(j["out"])
    out.mkdir(parents=True, exist_ok=True)
    name = j.get("name") or out.name
    size = parse_size(j["size"]) if j.get("size") else None
    refs = [upload(r) for r in j.get("refs") or []]
    if len(refs) > 10:
        sys.exit("FLUX 3 Klein takes at most 10 reference images")
    seed0 = j["seed"] if j.get("seed") is not None else random.randrange(2**32)

    queued = []
    for k in range(j["n"]):
        seed = seed0 + k
        pid = submit(build_graph(j["prompt"], seed, j["steps"], size, refs, f"flux3/{name}"))
        queued.append((seed, pid))

    written = []
    for seed, pid in queued:
        entry, waited = wait(pid)
        images = entry["outputs"]["save"]["images"]
        for img in images:
            dest = out / f"{name}_{seed}.png"
            fetch(img, dest)
            meta = {"model": UNET, "prompt": j["prompt"], "seed": seed, "steps": j["steps"],
                    "size": list(size) if size else "from first reference", "refs": j.get("refs") or [],
                    "sampler": "euler/simple", "cfg": 1.0, "seconds_waited": round(waited, 1),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}
            dest.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            line = f"{dest}  ({waited:.0f} s)"
            if j.get("nobg"):
                cut = dest.with_name(dest.stem + "_nobg.png")
                remove_background(dest, cut)
                line += f"\n{cut}"
            print(line, flush=True)
            written.append(dest)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="caption text (or use -f)")
    ap.add_argument("-f", "--prompt-file", help="read the caption from a UTF-8 file")
    ap.add_argument("-o", "--out", default="flux3-out", help="output folder (default ./flux3-out)")
    ap.add_argument("--name", help="file name prefix (default: output folder name)")
    ap.add_argument("-n", type=int, default=1, help="number of variants (consecutive seeds)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--size", help="WxH (multiple of 16) or ratio: " + ", ".join(RATIOS))
    ap.add_argument("--steps", type=int, default=16, help="distilled model: 16 (4-16 works)")
    ap.add_argument("--ref", action="append", default=[], help="reference image (repeat, up to 10)")
    ap.add_argument("--nobg", action="store_true", help="also write <name>_nobg.png with the background removed")
    ap.add_argument("--jobs", help="JSON list of jobs ({name, prompt, out, n, seed, size, steps, refs, nobg})")
    ap.add_argument("--keep", action="store_true", help="leave the server running afterwards")
    ap.add_argument("--stats", action="store_true", help="report peak VRAM and minimum free RAM")
    ap.add_argument("--stop", action="store_true", help="stop a server started earlier with --keep")
    a = ap.parse_args()

    if a.stop:
        stop_server()
        return

    if a.jobs:
        jobs = json.loads(Path(a.jobs).read_text(encoding="utf-8"))
    else:
        prompt = Path(a.prompt_file).read_text(encoding="utf-8").strip() if a.prompt_file else a.prompt
        if not prompt:
            ap.error("give a prompt or -f FILE or --jobs FILE")
        jobs = [{"prompt": prompt}]
    defaults = {"out": a.out, "name": a.name, "n": a.n, "seed": a.seed, "size": a.size,
                "steps": a.steps, "refs": a.ref, "nobg": a.nobg}

    sampler = Sampler() if a.stats else None
    if sampler:
        sampler.start()
    proc = None if server_up() else start_server()
    t0 = time.time()
    try:
        total = sum(len(run_job(job, defaults)) for job in jobs)
        print(f"{total} image(s) in {time.time() - t0:.0f} s", flush=True)
    finally:
        if proc and not a.keep:
            stop_server(proc)
        if sampler:
            print(sampler.stop())


if __name__ == "__main__":
    main()
