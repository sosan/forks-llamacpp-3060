# Elegir una de las dos listas según la GPU instalada:
#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# GTX 1650
# NCMOE_VALUES=[40, 39, 38, 37]
# RTX 3060
NCMOE_VALUES = [40, 24]
RUNS = 3

def main():
    parser = argparse.ArgumentParser(description="Benchmark llama-bench con muestreo de VRAM")
    parser.add_argument("--stock", default=os.environ.get("STOCK"),
                        help="Directorio que contiene llama-bench (o variable de entorno STOCK)")
    parser.add_argument("--model", default=os.environ.get("MODEL"),
                        help="Ruta al modelo GGUF (o variable de entorno MODEL)")
    parser.add_argument("--ncmoe", type=int, nargs="+", default=NCMOE_VALUES,
                        help=f"Valores de ncmoe (por defecto {NCMOE_VALUES})")
    parser.add_argument("--runs", type=int, default=RUNS,
                        help=f"Repeticiones por valor (por defecto {RUNS})")
    parser.add_argument("--outdir", default="results/stock-bench",
                        help="Directorio de salida")
    args = parser.parse_args()

    if not args.stock or not args.model:
        parser.error("Debes indicar --stock y --model (o definir STOCK y MODEL en el entorno)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # En Windows el ejecutable lleva extensión .exe
    exe_name = "llama-bench.exe" if sys.platform == "win32" else "llama-bench"
    llama_bench = Path(args.stock) / exe_name

    for ncmoe in args.ncmoe:
        for run in range(1, args.runs + 1):
            prefix = outdir / f"ncmoe{ncmoe}_run{run}"

            with open(f"{prefix}_vram_mib.log", "w") as vram_log:
                sampler = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        "-lms", "500",
                    ],
                    stdout=vram_log,
                    stderr=subprocess.DEVNULL,
                )

                # Dar tiempo al muestreador para iniciar la primera lectura.
                time.sleep(1)

                try:
                    with open(f"{prefix}_bench.json", "w") as out, \
                         open(f"{prefix}_bench.stderr", "w") as err:
                        result = subprocess.run(
                            [
                                str(llama_bench),
                                "-m", args.model,
                                "-ngl", "99",
                                "-ncmoe", str(ncmoe),
                                "-fa", "1",
                                "-p", "0",
                                "-n", "128",
                                "-d", "0,4096",
                                "-r", "3",
                                "-t", "4",
                                "-lm", "mmap",
                                "-o", "json",
                            ],
                            stdout=out,
                            stderr=err,
                        )
                    status = result.returncode
                except FileNotFoundError:
                    status = 127
                    print(f"No se encontró el ejecutable: {llama_bench}", file=sys.stderr)
                finally:
                    sampler.terminate()
                    try:
                        sampler.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        sampler.kill()
                        sampler.wait()

            line = f"ncmoe={ncmoe} run={run} exit={status}"
            print(line)  # equivalente a `tee`
            Path(f"{prefix}_status.txt").write_text(line + "\n")


if __name__ == "__main__":
    main()