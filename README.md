# flux3-imagegen

Local image generation and editing for coding agents with
[FLUX 3 Image Klein](https://huggingface.co/collections/black-forest-labs/flux-3-image-klein) (distilled, 5.6B)
through a headless ComfyUI. Runs on an 8 GB laptop GPU.

The agent writes the caption, generates, **looks at the results** and iterates — the same job BFL's 9B
prompt upsampler does, done by the agent instead, so the upsampler's ~10 GB stays out of memory.

| Part | What it is |
|---|---|
| `skills/imagegen` | Claude Code skill: when to use it, prompt guide condensed from BFL's upsampler system prompt, review loop |
| `mcp_server.py` | MCP tools `generate_image` and `generate_batch` for any MCP client |
| `flux3.py` | CLI: text-to-image, edits with up to 10 reference images, batches, background removal, reproducible sidecars |

Measured on RTX 4060 Laptop 8 GB / 16 GB RAM: 13–22 s per 1 MP image, 28–34 s with a reference, peak VRAM 7.3 GB.

## Install

Needs [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with 8 GB+.

1. **ComfyUI ≥ v0.41** (FLUX 3 support landed in v0.41.0). On Windows the
   [portable build](https://github.com/Comfy-Org/ComfyUI/releases) works out of the box; update an older one
   with `update\update_comfyui_stable.bat`. Any other ComfyUI install works if you start it yourself on port 8188.
2. **Weights** (gated: accept the license on
   [flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy), then `hf auth login`):
   ```
   hf download black-forest-labs/flux-3-image-klein-comfy --local-dir weights ^
     diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors ^
     text_encoders/qwen3vl_4b_int8_convrot.safetensors vae/flux2-vae.safetensors
   ```
   Move each file into the matching `ComfyUI/models/<folder>/`. Use the `int8_convrot` files on RTX 30/40;
   the NVFP4 ones need Blackwell.
3. **Plugin** (Claude Code):
   ```
   /plugin install flux3-imagegen --marketplace nazariipolishchuk/flux3-imagegen
   ```
   When asked, point *ComfyUI portable folder* at your `ComfyUI_windows_portable` (or leave it empty if you
   run ComfyUI yourself).

Other MCP clients: run `uv run --script mcp_server.py` with `FLUX3_COMFY_DIR` set.

## CLI

```
uv run --script flux3.py "Studio photograph, a matte black ceramic teapot ..." -o out/teapot -n 4
uv run --script flux3.py -f caption.txt --ref style_a.png --ref style_b.png --nobg -o out/icons
uv run --script flux3.py --jobs jobs.json --stats
```

`--keep` leaves the server running between calls; `--stop` stops it.

## Licenses

Code: MIT. Not affiliated with Black Forest Labs. Model weights are **not** included and must not be
redistributed: FLUX 3 Image Klein is under the FLUX Kommunity License v1.0 (outputs may be used commercially
by organisations under US$5M annual revenue; read the license for your case). The Qwen3-VL-4B text encoder
is Apache-2.0.
