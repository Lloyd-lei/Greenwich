"""AlphaMotion CLI: serve / download / eval / bench / ingest-urdf / doctor."""
from __future__ import annotations

import json
import platform
import sys

import typer

app = typer.Typer(add_completion=False, help="AlphaMotion motion engine")


@app.command()
def doctor():
    """Self-check: platform, GPU, GL, ffmpeg, weights, DB, ports."""
    from . import __version__
    from .config import CONFIG
    from .paths import cache_dir, data_dir, db_path
    rows: list[tuple[str, str]] = [
        ("alphamotion", __version__),
        ("platform", f"{platform.system()} {platform.release()}"),
        ("python", sys.version.split()[0]),
    ]
    try:
        import torch
        rows.append(("torch", f"{torch.__version__} cuda={torch.cuda.is_available()}"))
    except Exception as e:
        rows.append(("torch", f"FAIL {e}"))
    try:
        import mujoco
        rows.append(("mujoco", mujoco.__version__))
    except Exception as e:
        rows.append(("mujoco", f"FAIL {e}"))
    try:
        import imageio_ffmpeg
        rows.append(("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception as e:
        rows.append(("ffmpeg", f"FAIL {e}"))
    from .weights import status
    for k, ok in status().items():
        rows.append((f"weights/{k}", "ok" if ok else "not downloaded"))
    rows.append(("cache", str(cache_dir())))
    rows.append(("data", str(data_dir())))
    rows.append(("db", str(db_path())))
    rows.append(("device", CONFIG.device))
    width = max(len(k) for k, _ in rows)
    bad = 0
    for k, v in rows:
        flag = "!!" if "FAIL" in str(v) else "  "
        bad += "FAIL" in str(v)
        typer.echo(f" {flag} {k:<{width}}  {v}")
    raise typer.Exit(1 if bad else 0)


@app.command()
def download(third_party: bool = typer.Option(False, "--third-party",
                                              help="also fetch GENMO/T5")):
    """Fetch model weights from the HF hub (first run)."""
    from .weights import download_all
    out = download_all()
    for k, v in out.items():
        typer.echo(f"  {k}: {v}")
    if third_party:
        typer.echo("third-party perception weights are fetched by the GENMO "
                   "adapter on first use (see docs/SDK.md)")


@app.command()
def serve(host: str = "", port: int = 0):
    """Start the AlphaMotion service (API + frontend + viewers)."""
    import uvicorn

    from .config import CONFIG
    from .service.app import create_app
    uvicorn.run(create_app(),
                host=host or CONFIG.host, port=port or CONFIG.port)


@app.command("ingest-urdf")
def ingest_urdf(path: str, name: str = ""):
    """Parse + check + register a user URDF (offline entry)."""
    from .embodiment.urdf_ingest import ingest
    report = ingest(path, name or None)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@app.command()
def eval(out: str = "docs/BENCHMARK.md"):
    """Run the full release gate and write the benchmark table."""
    from .utils.eval_gate import run_gate
    ok = run_gate(out)
    raise typer.Exit(0 if ok else 1)


@app.command()
def bench():
    """Latency of the warm pool (encode/decode/bridge)."""
    from .utils.eval_gate import run_bench
    run_bench()


def main():
    app()


if __name__ == "__main__":
    main()
