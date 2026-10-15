---
name: imagegen
description: Use when the user asks to generate, draw or edit an image, art, illustration, icon, sprite, texture, concept, poster, mockup picture, game or app asset ("generate an image", "draw", "make an icon", "same style as", "згенеруй картинку", "намалюй", "зроби іконку", "асет", "арт", "в тому ж стилі"), or when a task needs a raster picture that does not exist yet.
---

# Image generation with local FLUX 3 Image Klein

You write the prompt (you replace BFL's 9B prompt upsampler), run the generator, **look at every result**
(Read the PNG), and iterate. The user gets chosen results, not raw first attempts.

## 1. Pick the approach

| Need | Approach |
|---|---|
| Art, illustration, game/app asset, product shot, poster with short text | FLUX 3 Klein (this plugin) |
| Same style as existing assets, edits ("make it red", "replace X"), more views of one object | FLUX 3 Klein with `refs` (up to 10) |
| Simple UI icon, logo, glyph, flat shape that must stay crisp | write SVG by hand instead |

## 2. Run

Preferred: the MCP tools `generate_image` (one prompt, `n` variants) and `generate_batch` (many assets in one
server session, models load once). CLI equivalent (long prompts go in a UTF-8 file):

```
FLUX3_COMFY_DIR="${user_config.comfy_dir}" uv run --script "${CLAUDE_PLUGIN_ROOT}/flux3.py" -f prompt.txt -o <out> -n 4 [--size 16:9] [--ref a.png ...] [--nobg] [--stats]
```

- The ComfyUI server starts on demand (25-45 s) and stops after the batch; an already running ComfyUI on
  port 8188 is reused. Expect ~15-35 s per 1 MP image on an 8 GB laptop GPU.
- Output: `<name>_<seed>.png` + `.json` sidecar (prompt, seed, settings). Same prompt + seed = same image.
- `nobg` / `--nobg` adds `_nobg.png` (rembg isnet, trimmed). FLUX never outputs alpha itself.
- Sizes: multiples of 16, ~1 MP. Ratios: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16. With refs and no size the
  output follows the first reference's shape.

## 3. Write the prompt (BFL upsampler rules, condensed)

One continuous English prose caption, 100-350 words, the exact text the model receives:
1. **Lead with medium and style**: "Studio photograph, ...", "Flat vector illustration, ...", "Stylized 3D
   game asset render, hand-painted texture look, ...", "Pixel art sprite, ...". No stated style → a photograph.
2. Then scene, lighting, camera; then subjects with **concrete colors, materials, state**.
3. **Layout in words**, not coordinates: frame-left / frame-right / upper / lower / foreground / midground /
   background, sentence-initial anchors ("On the left, ..."), relations (holds, faces, behind). Up to ~10
   instances one by one; beyond that give the exact count and arrangement. Bounding boxes work this way too:
   BFL's upsampler turns boxes into these words; the image model never sees coordinates.
4. **Be definite**: no hedging, no "beautiful/stunning", no closing mood sentence; turn moods into physical
   facts (light, color, posture). State negations as what IS present ("plain white background").
5. **Text in the image**: quote it verbatim in its language, give typography, color and placement.
6. People in focus: ethnicity, gender, age range, skin tone, hair, clothing.
7. Background ≈ 1/6 of the caption; distant planes are shapes and colors, never readable text.

Presets to append:
- Isolated asset: "centered on a plain pure white background, no floor, no cast shadow, even soft studio
  lighting, entire object visible with clear margin" → then `nobg`.
- Consistent set: make one hero asset, get it approved, then generate the rest with `refs: [hero]` and
  "in exactly the same style, palette, outline softness and lighting as the reference".
- Edits with refs: short instructions work ("change the dress color to red"); list what must stay
  ("keep the shape, camera angle, lighting and background exactly as they are").

## 4. Review loop (mandatory)

1. New idea → `n: 4` (consecutive seeds).
2. **Read every PNG.** Reject wrong counts, broken hands or letters, off-style, cropped subject, artifacts.
   Known weak spots: small counts ("5 stars" → a dozen) and small object states ("unlatched" → latched).
3. Fix the prompt where it failed (be more concrete), or keep the good seed and edit it with `refs`.
4. Report rejected variants in one line each; give the seeds of the delivered files.

## 5. Project styles

Keep a project's look in the project itself: `<project>/art/style.md` (caption fragment: medium, palette,
line weight, lighting) and `<project>/art/refs/` (2-3 approved images passed as `refs`). Read them before
generating for that project; add a result to `refs/` only after the user approves it.

## Facts

- FLUX 3 Image Klein distilled 5.6B, int8 convrot (5.8 GB) + Qwen3-VL-4B int8 text encoder (4.9 GB) + flux2
  VAE; 16 steps, cfg 1, euler/simple; needs ComfyUI ≥ v0.41.0.
- Measured on RTX 4060 Laptop 8 GB / 16 GB RAM: t2i 13-22 s, with one ref 28-34 s, peak VRAM 7.3 GB, free RAM
  down to ~1 GB with everyday apps open - suggest closing the browser for long batches.
- License: FLUX Kommunity v1.0 - outputs may be used commercially by organisations under US$5M revenue;
  weights must not be redistributed.
