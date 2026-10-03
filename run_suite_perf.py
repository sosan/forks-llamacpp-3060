#!/usr/bin/env python3
"""
Suite de benchmarks para fork-perf
Basado en Sección 4 del documento
"""

import subprocess
import sys

# Configuraciones para fork-perf
#
# El contexto cambia cuántos expertos caben (≈124/capa a 4K y ≈88/capa a 64K
# en la RTX 3060), así que la suite barre ambos contextos: 64K es la referencia
# del artículo para este fork y 4K permite compararlo con el fork moe-cache.
PERF_CONFIGS = [
    # --- Contexto 64K (referencia del artículo) ---
    {
        "mode": "both",
        "slots": 64,
        "ncmoe": 99,
        "ctx": 65536,
        "mtp": False,
        "desc": "64K sin MTP: 64 slots, ncmoe 99"
    },
    {
        "mode": "both",
        "slots": 88,
        "ncmoe": 99,
        "ctx": 65536,
        "mtp": False,
        "desc": "64K sin MTP: 88 slots, ncmoe 99"
    },
    {
        "mode": "both",
        "slots": 64,
        "ncmoe": 99,
        "ctx": 65536,
        "mtp": True,
        "desc": "64K con MTP: 64 slots, ncmoe 99"
    },
    {
        "mode": "both",
        "slots": 88,
        "ncmoe": 99,
        "ctx": 65536,
        "mtp": True,
        "desc": "64K con MTP: 88 slots, ncmoe 99"
    },

    # --- Contexto 4K (comparable con el fork moe-cache) ---
    {
        "mode": "both",
        "slots": 88,
        "ncmoe": 99,
        "ctx": 4096,
        "mtp": False,
        "desc": "4K sin MTP: 88 slots, ncmoe 99"
    },
    {
        "mode": "both",
        "slots": 88,
        "ncmoe": 99,
        "ctx": 4096,
        "mtp": True,
        "desc": "4K con MTP: 88 slots, ncmoe 99"
    },

    # --- Con capas en GPU (si cabe en VRAM) ---
    {
        "mode": "both",
        "slots": 64,
        "ncmoe": 24,
        "ctx": 4096,
        "mtp": False,
        "desc": "4K con GPU: ncmoe 24"
    },
    {
        "mode": "both",
        "slots": 64,
        "ncmoe": 24,
        "ctx": 4096,
        "mtp": True,
        "desc": "4K con GPU + MTP: ncmoe 24"
    },
]


def main():
    print("🚀 Suite fork-perf")
    print(f"   Configuraciones: {len(PERF_CONFIGS)}")

    results = []
    for idx, config in enumerate(PERF_CONFIGS):
        desc = config["desc"]
        print(f"\n{'='*70}")
        print(f"🧪 [{idx+1}/{len(PERF_CONFIGS)}] {desc}")
        print(f"{'='*70}")

        cmd = ["python", "test-perf.py", config["mode"]]

        if "slots" in config:
            cmd += ["--slots", str(config["slots"])]
        if "ncmoe" in config:
            cmd += ["--ncmoe", str(config["ncmoe"])]
        if "ctx" in config:
            cmd += ["--ctx", str(config["ctx"])]
        if config.get("mtp"):
            cmd.append("--mtp")

        print(f"💻 {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd)
            ok = result.returncode == 0
            print(f"\n{'✅' if ok else '❌'} {desc}")
            results.append((desc, ok))
        except KeyboardInterrupt:
            print("\n⚠️  Interrumpido")
            sys.exit(1)

    # Resumen
    print(f"\n{'='*70}")
    print("📊 RESUMEN")
    print(f"{'='*70}")

    for desc, ok in results:
        print(f"{'✅' if ok else '❌'} {desc}")

    successful = sum(1 for _, ok in results if ok)
    print(f"\nTotal: {successful}/{len(results)} exitosas")


if __name__ == "__main__":
    main()