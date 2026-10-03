#!/usr/bin/env python3
"""
Benchmark de llama-server para llama.cpp STOCK (sin fork).
Compatible con Linux y Windows. Solo usa la librería estándar.

Ejemplos:
  python test-stock.py
  python test-stock.py --ncmoe 40 24 --runs 3 --ctx 4096
  python test-stock.py --bin-dir C:\\llama.cpp\\build\\bin --model C:\\modelos\\m.gguf --prompt p.txt
  python test-stock.py --extra-args "--no-mmap"
"""

import argparse
import csv
import json
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"

# Valores por defecto (mismos que el script del fork)
DEFAULT_WORKDIR = Path("/workspace")
DEFAULT_BIN_DIR = DEFAULT_WORKDIR / "llama.cpp" / "build" / "bin"
DEFAULT_MODEL = DEFAULT_WORKDIR / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DEFAULT_PROMPT = DEFAULT_WORKDIR / "prompt2434.txt"


# --------------------------------------------------------------------------- #
# Utilidades de procesos / puertos
# --------------------------------------------------------------------------- #
def kill_process_tree(proc):
    """Mata un proceso (y sus hijos) de forma multiplataforma."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, check=False)
        else:
            import os
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)
    except Exception as e:
        print(f"Aviso: no se pudo matar el proceso {proc.pid}: {e}")


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_server_on_port(port):
    """Mata lo que esté escuchando en el puerto (netstat en Windows, lsof en Linux)."""
    try:
        if IS_WINDOWS:
            out = subprocess.run(["netstat", "-ano"], capture_output=True,
                                 text=True, check=False).stdout
            pids = {line.split()[-1] for line in out.splitlines()
                    if f":{port} " in line and "LISTENING" in line}
            for pid in pids:
                print(f"Matando proceso {pid} en puerto {port}")
                subprocess.run(["taskkill", "/F", "/PID", pid],
                               capture_output=True, check=False)
        else:
            out = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True,
                                 text=True, check=False).stdout
            for pid in out.split():
                print(f"Matando proceso {pid} en puerto {port}")
                subprocess.run(["kill", "-9", pid], capture_output=True, check=False)
    except FileNotFoundError as e:
        print(f"Aviso: herramienta no disponible para liberar el puerto: {e}")
    time.sleep(2)


# --------------------------------------------------------------------------- #
# Muestreo de VRAM
# --------------------------------------------------------------------------- #
def start_vram_sampler(log_path):
    """Lanza nvidia-smi en bucle (cada 500 ms). Devuelve (proc, fichero) o (None, None)."""
    try:
        f = open(log_path, "w")
        proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-lms", "500"],
            stdout=f, stderr=subprocess.DEVNULL,
            start_new_session=not IS_WINDOWS,
        )
        return proc, f
    except FileNotFoundError:
        print("Aviso: nvidia-smi no encontrado; se omite el muestreo de VRAM")
        return None, None


def max_vram(log_path):
    best = 0
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    best = max(best, int(line))
    except FileNotFoundError:
        pass
    return best


# --------------------------------------------------------------------------- #
# Servidor
# --------------------------------------------------------------------------- #
def build_server_command(args, ncmoe):
    exe = "llama-server.exe" if IS_WINDOWS else "llama-server"
    cmd = [
        str(Path(args.bin_dir) / exe),
        "-m", str(args.model),
        "-c", str(args.ctx), "-b", str(args.batch), "-ub", str(args.ubatch),
        "-np", "1", "-n", str(args.n_predict),
        "--no-warmup", "-fa", "on",
        "-t", str(args.threads),
        "-ctk", "f16", "-ctv", "f16", "-kvo",
        "--fit", "off", "--jinja", "--reasoning", "off",
        "-ngl", str(args.ngl),
        "--n-cpu-moe", str(ncmoe),
        "--cache-ram", "0", "--spec-type", "none", "--log-colors", "off",
        "--port", str(args.port), "--host", args.host,
    ]
    if args.load_mode:
        cmd += ["--load-mode", args.load_mode]
    if args.extra_args:
        cmd += shlex.split(args.extra_args, posix=not IS_WINDOWS)
    return cmd


def wait_for_server(port, server, timeout):
    start = time.time()
    while time.time() - start < timeout:
        if server.poll() is not None:  # el servidor murió
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def send_request(args, prompt_text):
    body = {
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0,
        "max_tokens": args.gen_tokens,
        "stream": False,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.request_timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def print_tail(path, n=50):
    print(f"\nÚltimas {n} líneas del log:")
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f.readlines()[-n:]:
                print(line.rstrip())
    except FileNotFoundError:
        print("(log no encontrado)")


# --------------------------------------------------------------------------- #
# Una ejecución
# --------------------------------------------------------------------------- #
def run_test(args, ncmoe, run, prompt_text, outdir):
    prefix = outdir / f"stock_ncmoe{ncmoe}_c{args.ctx}_run{run}"
    log_path = Path(f"{prefix}.log")
    vram_path = Path(f"{prefix}_vram_mib.log")

    print("\n" + "=" * 50)
    print(f"llama.cpp stock | ncmoe={ncmoe} ctx={args.ctx} run={run}/{args.runs}")
    print("=" * 50)

    if port_in_use(args.port):
        if args.kill_port:
            kill_server_on_port(args.port)
        if port_in_use(args.port):
            print(f"ERROR: puerto {args.port} ocupado (usa --kill-port o cambia --port)")
            sys.exit(1)

    sampler, sampler_file = start_vram_sampler(vram_path)
    time.sleep(1)  # dejar que el muestreador haga la primera lectura

    cmd = build_server_command(args, ncmoe)
    if args.verbose:
        print("Comando:", " ".join(cmd))

    result = {"ncmoe": ncmoe, "ctx": args.ctx, "run": run, "ok": False}
    server = None
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        try:
            server = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT,
                start_new_session=not IS_WINDOWS,
            )
        except FileNotFoundError:
            print(f"ERROR: no se encontró el ejecutable: {cmd[0]}")
            sys.exit(1)

        print("Esperando a que el servidor esté listo...")
        if not wait_for_server(args.port, server, args.startup_timeout):
            print("ERROR: el servidor no arrancó (timeout o proceso terminado)")
            log_file.flush()
            print_tail(log_path)
            return result
        print("Servidor listo. Enviando petición...")

        try:
            response = send_request(args, prompt_text)
        except Exception as e:
            print(f"ERROR: la petición falló: {e}")
            return result

        Path(f"{prefix}_response.json").write_text(
            json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")

        t = response.get("timings", {})
        if t:
            result.update(
                prompt_n=t.get("prompt_n"),
                prompt_ms=t.get("prompt_ms"),
                prompt_tps=t.get("prompt_per_second"),
                decode_n=t.get("predicted_n"),
                decode_ms=t.get("predicted_ms"),
                decode_tps=t.get("predicted_per_second"),
            )
        result["ok"] = True
    finally:
        kill_process_tree(server)
        kill_process_tree(sampler)
        log_file.close()
        if sampler_file:
            sampler_file.close()
        time.sleep(2)

    result["vram_max_mib"] = max_vram(vram_path)

    print(f"VRAM máxima: {result['vram_max_mib']} MiB")
    if "prompt_tps" in result:
        print(f"  prompt: {result['prompt_n']} tokens en {result['prompt_ms']:.2f} ms "
              f"({result['prompt_tps']:.2f} t/s)")
        print(f"  decode: {result['decode_n']} tokens en {result['decode_ms']:.2f} ms "
              f"({result['decode_tps']:.2f} t/s)")
    else:
        print("  (la respuesta no incluye 'timings')")
    print(f"Logs: {log_path}, {vram_path}")
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Benchmark de llama-server (llama.cpp stock)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR,
                   help="Directorio con llama-server")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Modelo GGUF")
    p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT, help="Fichero de prompt")
    p.add_argument("--outdir", type=Path, default=Path("results/stock-server"),
                   help="Directorio de salida")
    p.add_argument("--ncmoe", type=int, nargs="+", default=[24],
                   help="Valores de --n-cpu-moe a probar")
    p.add_argument("--runs", type=int, default=1, help="Repeticiones por valor")
    p.add_argument("--ctx", type=int, default=4096, help="Tamaño de contexto (-c)")
    p.add_argument("--batch", type=int, default=2048, help="-b")
    p.add_argument("--ubatch", type=int, default=512, help="-ub")
    p.add_argument("--ngl", type=int, default=99, help="-ngl")
    p.add_argument("--threads", type=int, default=12, help="-t")
    p.add_argument("--n-predict", type=int, default=1024, help="-n del servidor")
    p.add_argument("--gen-tokens", type=int, default=128,
                   help="max_tokens de la petición")
    p.add_argument("--load-mode", default="none",
                   help="--load-mode del servidor (cadena vacía para omitirlo)")
    p.add_argument("--extra-args", default="",
                   help='Argumentos extra para llama-server, p. ej. "--no-mmap"')
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8189)
    p.add_argument("--kill-port", action="store_true",
                   help="Matar lo que ocupe el puerto antes de empezar")
    p.add_argument("--startup-timeout", type=int, default=300)
    p.add_argument("--request-timeout", type=int, default=300)
    p.add_argument("--verbose", action="store_true", help="Mostrar el comando lanzado")
    args = p.parse_args()

    if not args.model.exists():
        sys.exit(f"ERROR: modelo no encontrado: {args.model}")
    if not args.prompt.exists():
        sys.exit(f"ERROR: prompt no encontrado: {args.prompt}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    prompt_text = args.prompt.read_text(encoding="utf-8")

    results = []
    for ncmoe in args.ncmoe:
        for run in range(1, args.runs + 1):
            results.append(run_test(args, ncmoe, run, prompt_text, args.outdir))

    # Resumen
    fields = ["ncmoe", "ctx", "run", "ok", "vram_max_mib",
              "prompt_n", "prompt_ms", "prompt_tps",
              "decode_n", "decode_ms", "decode_tps"]
    csv_path = args.outdir / f"summary_c{args.ctx}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 50)
    print("Resumen")
    print("=" * 50)
    print(f"{'ncmoe':>5} {'run':>3} {'VRAM MiB':>9} {'prompt t/s':>11} {'decode t/s':>11}")
    for r in results:
        if r["ok"]:
            print(f"{r['ncmoe']:>5} {r['run']:>3} {r['vram_max_mib']:>9} "
                  f"{r.get('prompt_tps') or 0:>11.2f} {r.get('decode_tps') or 0:>11.2f}")
        else:
            print(f"{r['ncmoe']:>5} {r['run']:>3} {'FALLÓ':>9}")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()