# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=2,<3", "rembg[cpu]>=2.0.60", "pillow"]
# ///
"""
MCP server exposing local FLUX 3 Klein generation to any MCP-capable agent.

    uv run --script mcp_server.py            (stdio)

It shells out to flux3.py, so server lifecycle, sidecars and --nobg behave exactly as on the CLI.
Set FLUX3_COMFY_DIR to the ComfyUI_windows_portable folder, or run your own ComfyUI on FLUX3_PORT.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp.server.mcpserver import MCPServer  # mcp 2.x (FastMCP was renamed)

ROOT = Path(__file__).resolve().parent
mcp = MCPServer("flux3")


def _run(args: list[str]) -> str:
    run = subprocess.run([sys.executable, str(ROOT / "flux3.py"), *args],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    if run.returncode:
        return f"FAILED (exit {run.returncode}):\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}"
    return run.stdout


@mcp.tool()
def generate_image(prompt: str, out_dir: str, n: int = 1, size: str | None = None, seed: int | None = None,
                   refs: list[str] | None = None, nobg: bool = False, name: str | None = None) -> str:
    """Generate images locally with FLUX 3 Image Klein (16 steps, ~1 MP, ~15-35 s each on an 8 GB GPU).

    prompt: one English prose caption - medium and style first, then scene, lighting, subjects with
            concrete colors/materials, layout in frame-relative words (frame-left, foreground...).
            Quote any text to render verbatim. No coordinates, no negative-prompt lists.
    out_dir: folder for PNGs (each gets a .json sidecar with prompt and seed).
    n: variants on consecutive seeds. size: "WxH" (multiple of 16) or 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16.
    refs: up to 10 reference images for edits or style consistency (the prompt then describes the change).
    nobg: also write *_nobg.png with the background removed and trimmed.
    Returns the written file paths and timings. Look at the images before using them.
    """
    args = [prompt, "-o", out_dir, "-n", str(n)]
    for flag, value in (("--size", size), ("--seed", seed), ("--name", name)):
        if value is not None:
            args += [flag, str(value)]
    for r in refs or []:
        args += ["--ref", r]
    if nobg:
        args.append("--nobg")
    return _run(args)


@mcp.tool()
def generate_batch(jobs: list[dict]) -> str:
    """Generate several different assets in one server session (models load once).

    jobs: list of objects with keys prompt (required), out (folder, required), name, n, seed, size,
          refs (list of paths), nobg (bool) - same meaning as in generate_image.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False)
    try:
        return _run(["--jobs", f.name])
    finally:
        Path(f.name).unlink(missing_ok=True)


if __name__ == "__main__":
    mcp.run()
