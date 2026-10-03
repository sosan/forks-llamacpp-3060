#!/usr/bin/env python3
"""
Perfilado y servicio para el fork `perf` de llama.cpp (caché estática de expertos).

Reproduce en Python los dos pasos del artículo:
  1. trace:  genera dos trazas de expertos (código y charla) con `llama-moe-trace`
             y las fusiona en un único CSV.
  2. serve:  arranca `llama-server` con el perfil fusionado, huecos por capa y MTP.

Basado en la sección 4 del documento de comandos. Solo librería estándar.

Uso:
  python run_traces.py trace --bin-dir "$PERF" --model "$MODEL" --outdir traces
  python run_traces.py serve --bin-dir "$PERF" --model "$MODEL" --profile traces/qwen35b-merged.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Prompts por defecto (los mismos del artículo)
PROMPT_CODE = (
    "Write a Python implementation of an LRU cache with O(1) operations, "
    "then explain the design choices in detail. After that, write the same "
    "thing in Rust and compare the two implementations."
)
PROMPT_CHAT = (
    "Write a warm, personal letter to a friend describing a memorable ibiza "
    "summer, with vivid sensory details. Then reflect on why travel memories "
    "stay with us longer than everyday ones."
)

DEFAULTS = {
    "bin_dir": Path("/workspace/fork-perf/build/bin"),
    "model": Path("/workspace/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    "outdir": Path("traces"),
    "ngl": 99,
    "ncmoe": 99,
    "ctx": 4096,
    "n_predict": 512,
    "port": 8189,
    "host": "0.0.0.0",
    "slots": 88,
    "serve_ctx": 65536,
    "spec_draft_n_max": 2,
    "temp": 0.7,
    "top_k": 20,
    "top_p": 0.95,
}


def binary(bin_dir: Path, name: str) -> str:
    """Ruta a un binario del fork, con extensión .exe en Windows."""
    import platform
    path = bin_dir / name
    if platform.system() == "Windows":
        path = path.with_suffix(".exe")
    return str(path)


def check_inputs(args) -> None:
    if not Path(args.model).exists():
        print(f"❌ ERROR: Modelo no encontrado en {args.model}")
        sys.exit(1)


def run_trace(args) -> int:
    """Genera las trazas de código y charla y las fusiona."""
    bin_dir = Path(args.bin_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trace_bin = binary(bin_dir, "llama-moe-trace")
    code_csv = outdir / "qwen35b-code.csv"
    chat_csv = outdir / "qwen35b-chat.csv"
    merged_csv = outdir / "qwen35b-merged.csv"

    samples = [
        ("código", code_csv, args.prompt_code),
        ("charla", chat_csv, args.prompt_chat),
    ]

    for label, out_csv, prompt in samples:
        cmd = [
            trace_bin,
            "-m", str(args.model),
            "-ngl", str(args.ngl), "-ncmoe", str(args.ncmoe),
            "-fa", "1", "-c", str(args.ctx), "-n", str(args.n_predict),
            "-p", prompt,
        ]
        print(f"\n{'=' * 70}")
        print(f"🧪 Traza de {label} -> {out_csv}")
        print(f"{'=' * 70}")
        print(f"💻 MOE_TRACE_OUT={out_csv} {' '.join(cmd)}")

        env = dict(**__import__("os").environ, MOE_TRACE_OUT=str(out_csv))
        try:
            result = subprocess.run(cmd, env=env)
        except FileNotFoundError:
            print(f"❌ No se encontró el binario {trace_bin}")
            return 1
        if result.returncode != 0:
            print(f"❌ Falló la traza de {label} (código {result.returncode})")
            return result.returncode

    print(f"\n🧬 Fusionando trazas en {merged_csv}")
    with open(merged_csv, "wb") as out:
        for src in (code_csv, chat_csv):
            with open(src, "rb") as f:
                out.write(f.read())

    print(f"✅ Trazas listas: {code_csv}, {chat_csv} -> {merged_csv}")
    return 0


def run_serve(args) -> int:
    """Arranca llama-server con el perfil y los huecos por capa."""
    bin_dir = Path(args.bin_dir)
    server_bin = binary(bin_dir, "llama-server")
    profile = Path(args.profile)

    if not profile.exists():
        print(f"❌ ERROR: Perfil no encontrado en {profile}")
        return 1

    cmd = [
        server_bin,
        "-m", str(args.model),
        "--host", str(args.host), "--port", str(args.port),
        "--jinja", "--load-mode", args.load_mode,
        "-ngl", str(args.ngl), "-ncmoe", str(args.ncmoe), "-fa", "1",
        "-c", str(args.serve_ctx),
        "--moe-cache-profile", str(profile),
        "--moe-cache-slots", str(args.slots),
        "--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.spec_draft_n_max),
        "--temp", str(args.temp), "--top-k", str(args.top_k), "--top-p", str(args.top_p),
    ]

    print(f"\n{'=' * 70}")
    print(f"🚀 Servidor fork-perf: {args.slots} slots, c={args.serve_ctx}")
    print(f"{'=' * 70}")
    print(f"💻 {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd)
    except FileNotFoundError:
        print(f"❌ No se encontró el binario {server_bin}")
        return 1
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perfilado y servicio para el fork `perf` de llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s trace --bin-dir "$PERF" --model "$MODEL" --outdir traces
  %(prog)s serve --bin-dir "$PERF" --model "$MODEL" \\
      --profile traces/qwen35b-merged.csv --slots 88 --serve-ctx 65536
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--bin-dir", default=DEFAULTS["bin_dir"],
                       help="Directorio de binarios del fork perf")
        p.add_argument("--model", default=DEFAULTS["model"], help="Fichero GGUF")
        p.add_argument("--ngl", type=int, default=DEFAULTS["ngl"],
                       help="Capas en GPU (default: 99)")
        p.add_argument("--ncmoe", type=int, default=DEFAULTS["ncmoe"],
                       help="Capas MoE en CPU (default: 99)")
        p.add_argument("--outdir", default=DEFAULTS["outdir"],
                       help="Directorio de trazas/salida (default: traces)")

    # --- trace ---
    p_trace = sub.add_parser("trace", help="Genera y fusiona las trazas de expertos")
    add_common(p_trace)
    p_trace.add_argument("--ctx", type=int, default=DEFAULTS["ctx"],
                         help="Contexto del perfilado (default: 4096)")
    p_trace.add_argument("--n-predict", type=int, default=DEFAULTS["n_predict"],
                         help="Tokens a generar por pasada (default: 512)")
    p_trace.add_argument("--prompt-code", default=PROMPT_CODE,
                         help="Prompt de código para el perfilado")
    p_trace.add_argument("--prompt-chat", default=PROMPT_CHAT,
                         help="Prompt de conversación para el perfilado")
    p_trace.set_defaults(func=run_trace)

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Arranca llama-server con el perfil")
    add_common(p_serve)
    p_serve.add_argument("--profile", default="traces/qwen35b-merged.csv",
                         help="CSV del perfil fusionado")
    p_serve.add_argument("--slots", type=int, default=DEFAULTS["slots"],
                         help="Huecos de expertos por capa (default: 88)")
    p_serve.add_argument("--serve-ctx", type=int, default=DEFAULTS["serve_ctx"],
                         help="Contexto del servidor (default: 65536)")
    p_serve.add_argument("--host", default=DEFAULTS["host"], help="Host del servidor")
    p_serve.add_argument("--port", type=int, default=DEFAULTS["port"],
                         help="Puerto del servidor (default: 8189)")
    p_serve.add_argument("--load-mode", default="none", choices=["mmap", "none"],
                         help="Modo de carga (default: none)")
    p_serve.add_argument("--spec-draft-n-max", type=int,
                         default=DEFAULTS["spec_draft_n_max"],
                         help="Tokens de borrador MTP (default: 2)")
    p_serve.add_argument("--temp", type=float, default=DEFAULTS["temp"])
    p_serve.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    p_serve.add_argument("--top-p", type=float, default=DEFAULTS["top_p"])
    p_serve.set_defaults(func=run_serve)

    return parser


def main():
    args = build_parser().parse_args()
    check_inputs(args)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
