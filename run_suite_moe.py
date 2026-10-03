#!/usr/bin/env python3
"""
Suite parametrizable de benchmarks para el fork `moe-cache`.

Ejemplos:
  python run_suite_moe.py
  python run_suite_moe.py --mode all --slots 124 --ncmoe 20 --ctx 4096
  python run_suite_moe.py --slots 64 96 124 --ncmoe 20 --ctx 4096
  python run_suite_moe.py --only "oficial" --dry-run
  python run_suite_moe.py --smoke --stop-on-error
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

# El contexto cambia cuántos expertos caben (≈124/capa a 4K y ≈88/capa a 64K
# en la RTX 3060), así que la suite barre ambos contextos. A 4K la capacidad es
# mayor; a 64K se reduce y conviene comprobar que la caché sigue entrando.
MOECACHE_CONFIGS = [
    # --- Contexto 4K ---
    {
        "mode": "all", "slots": 124, "ncmoe": 20, "ctx": 4096,
        "desc": "4K capacidad: 124 slots, ncmoe 20",
    },
    {
        "mode": "all", "slots": 96, "ncmoe": 20, "ctx": 4096,
        "desc": "4K slots intermedios: 96, ncmoe 20",
    },
    {
        "mode": "all", "slots": 64, "ncmoe": 20, "ctx": 4096,
        "desc": "4K menos slots: 64, ncmoe 20",
    },
    {
        "mode": "all", "slots": 124, "ncmoe": 24, "ctx": 4096,
        "desc": "4K más GPU: 124 slots, ncmoe 24",
    },

    # --- Contexto 64K ---
    {
        "mode": "all", "slots": 88, "ncmoe": 20, "ctx": 65536,
        "desc": "64K capacidad: 88 slots, ncmoe 20",
    },
    {
        "mode": "all", "slots": 64, "ncmoe": 20, "ctx": 65536,
        "desc": "64K menos slots: 64, ncmoe 20",
    },
]


def configs_from_flags(args):
    """Genera el producto de slots, ncmoe y contexto solicitado."""
    slots = args.slots or [128]
    ncmoe = args.ncmoe or [20]
    ctx = args.ctx or [4096]
    mode = args.mode or "all"

    configs = []
    for slot, n_cpu, context in itertools.product(slots, ncmoe, ctx):
        configs.append({
            "mode": mode,
            "slots": slot,
            "ncmoe": n_cpu,
            "ctx": context,
            "desc": f"Parametrizada: {slot} slots, ncmoe {n_cpu}, c={context}",
        })
    return configs


def command_for(config, args):
    script = Path(__file__).with_name("test-moe.py")
    cmd = [
        sys.executable,
        str(script),
        config["mode"],
        "--slots", str(config["slots"]),
        "--ncmoe", str(config["ncmoe"]),
        "--ctx", str(config["ctx"]),
    ]
    if getattr(args, "threads", None) is not None:
        cmd.extend(["--threads", str(args.threads)])
    if getattr(args, "kv_type", None) is not None:
        cmd.extend(["--kv-type", args.kv_type])
    return cmd


def build_parser():
    parser = argparse.ArgumentParser(
        description="Suite parametrizable para el fork moe-cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s
  %(prog)s --mode all --slots 124 --ncmoe 20 --ctx 4096
  %(prog)s --slots 64 96 124 --ncmoe 20 --ctx 4096
  %(prog)s --only oficial --dry-run
        """,
    )
    parser.add_argument(
        "--mode", choices=["cache", "cache-only", "cache-full", "no-cache", "all"],
        help="Modo de test; por defecto, el definido en cada configuración",
    )
    parser.add_argument(
        "--slots", type=int, nargs="+",
        help="Uno o varios números de slots; activa el modo parametrizado",
    )
    parser.add_argument(
        "--ncmoe", type=int, nargs="+",
        help="Uno o varios valores de ncmoe; activa el modo parametrizado",
    )
    parser.add_argument(
        "--ctx", type=int, nargs="+",
        help="Uno o varios contextos; activa el modo parametrizado",
    )
    parser.add_argument(
        "--threads", type=int,
        help="Hilos CPU pasados a test-moe.py (default en test-moe.py: 4)",
    )
    parser.add_argument(
        "--kv-type", choices=["q8_0", "f16"],
        help="KV cache quant pasada a test-moe.py (default: q8_0)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Preflight: valida con cache-full --smoke-test cada combinación única "
             "(slots, ncmoe, ctx) antes de la suite. Aborta si alguna falla.",
    )
    parser.add_argument(
        "--vram-limit", type=int, default=11500,
        help="Límite VRAM en MiB para el smoke test (default: 11500)",
    )
    parser.add_argument(
        "--only",
        help="Filtrar las configuraciones por texto en su descripción",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Detener la suite tras el primer fallo",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostrar los comandos sin ejecutarlos",
    )
    return parser

def run_smoke(configs, args):
    """Preflight: valida cada combinación única (slots, ncmoe, ctx) con
    cache-full --smoke-test antes de la suite completa. Aborta si alguna falla."""
    script = Path(__file__).with_name("test-moe.py")
    unique = {(c["slots"], c["ncmoe"], c["ctx"]) for c in configs}
    seen = set()
    for config in configs:
        key = (config["slots"], config["ncmoe"], config["ctx"])
        if key in seen:
            continue
        seen.add(key)
        cmd = [
            sys.executable, str(script),
            "cache-full",
            "--slots", str(config["slots"]),
            "--ncmoe", str(config["ncmoe"]),
            "--ctx", str(config["ctx"]),
            "--smoke-test",
            "--vram-limit", str(args.vram_limit),
        ]
        if args.threads is not None:
            cmd.extend(["--threads", str(args.threads)])
        if args.kv_type is not None:
            cmd.extend(["--kv-type", args.kv_type])
        print(f"   [{len(seen)}/{len(unique)}] {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n❌ Smoke test falló; abortando suite")
            return False
    print("\n✓ Smoke tests superados; suite completa a continuación")
    return True


def main():
    args = build_parser().parse_args()

    parametrizada = any(value is not None for value in (args.slots, args.ncmoe, args.ctx, args.mode))
    configs = configs_from_flags(args) if parametrizada else list(MOECACHE_CONFIGS)

    if args.only:
        needle = args.only.casefold()
        configs = [c for c in configs if needle in c["desc"].casefold()]

    if not configs:
        print("No hay configuraciones que ejecutar.", file=sys.stderr)
        return 2

    print("🚀 Suite fork-moe-cache")
    print(f"   Configuraciones: {len(configs)}")

    if args.smoke:
        print("\n🔍 FASE DE SMOKE TEST (cache-full, n=32, preflight)")
        if not run_smoke(configs, args):
            return 1

    results = []
    for index, config in enumerate(configs, 1):
        cmd = command_for(config, args)
        print(f"\n{'=' * 70}")
        print(f"🧪 [{index}/{len(configs)}] {config['desc']}")
        print(f"{'=' * 70}")
        print(f"💻 {' '.join(cmd)}")

        if args.dry_run:
            results.append((config["desc"], True))
            continue

        try:
            result = subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n⚠️  Interrumpido")
            return 130

        ok = result.returncode == 0
        print(f"\n{'✅' if ok else '❌'} {config['desc']}")
        results.append((config["desc"], ok))
        if not ok and args.stop_on_error:
            print("\n🛑 Deteniendo suite")
            break

    print(f"\n{'=' * 70}")
    print("📊 RESUMEN")
    print(f"{'=' * 70}")
    for desc, ok in results:
        print(f"{'✅' if ok else '❌'} {desc}")

    successful = sum(1 for _, ok in results if ok)
    print(f"\nTotal: {successful}/{len(results)} exitosas")
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
