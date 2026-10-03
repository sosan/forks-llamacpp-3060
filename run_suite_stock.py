#!/usr/bin/env python3
"""
Suite de benchmarks para stock llama.cpp
Basado en Secciones 1-2 del documento de comandos
"""

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict


# ============================================================
# Rutas y configuración
# ============================================================

WORKDIR = Path("/workspace")
MODEL = WORKDIR / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
PROMPT = WORKDIR / "prompt2434.txt"
STOCK = WORKDIR / "stock" / "build" / "bin"


# ============================================================
# Configuraciones para stock (Secciones 1-2)
# ============================================================

# Sección 1: llama-bench barrido de ncmoe
STOCK_BENCH_CONFIGS = [
    # RTX 3060
    {
        "ncmoe": 40,
        "load_mode": "mmap",
        "desc": "RTX 3060: ncmoe 40 (0 capas GPU) mmap"
    },
    {
        "ncmoe": 24,
        "load_mode": "mmap",
        "desc": "RTX 3060: ncmoe 24 (16 capas GPU) mmap"
    },
    {
        "ncmoe": 40,
        "load_mode": "none",
        "desc": "RTX 3060: ncmoe 40 (0 capas GPU) none"
    },
    {
        "ncmoe": 24,
        "load_mode": "none",
        "desc": "RTX 3060: ncmoe 24 (16 capas GPU) none"
    },
]

# Sección 2: servidor con diferentes configuraciones
STOCK_SERVER_CONFIGS = [
    {
        "ncmoe": 40,
        "load_mode": "mmap",
        "ctx": 4096,
        "threads": 4,
        "desc": "Servidor: ncmoe 40, mmap, c=4096"
    },
    {
        "ncmoe": 40,
        "load_mode": "none",
        "ctx": 4096,
        "threads": 4,
        "desc": "Servidor: ncmoe 40, none, c=4096"
    },
    {
        "ncmoe": 24,
        "load_mode": "mmap",
        "ctx": 4096,
        "threads": 4,
        "desc": "Servidor: ncmoe 24, mmap, c=4096"
    },
    {
        "ncmoe": 24,
        "load_mode": "none",
        "ctx": 4096,
        "threads": 4,
        "desc": "Servidor: ncmoe 24, none, c=4096"
    },
]


# ============================================================
# Utilidades de proceso
# ============================================================

def _binary(name: str) -> str:
    """Ruta al binario de llama.cpp, con extensión en Windows."""
    path = STOCK / name
    if platform.system() == "Windows":
        path = path.with_suffix(".exe")
    return str(path)


def kill_process_tree(pid: int):
    """Mata un proceso y todos sus hijos."""
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(1)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception as e:
        print(f"Warning: No se pudo matar proceso {pid}: {e}")


def check_port_in_use(port: int) -> bool:
    """Verifica si un puerto está en uso."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def kill_server_on_port(port: int):
    """Mata cualquier servidor en el puerto especificado."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, check=False
            )
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = int(parts[-1])
                        print(f"Matando proceso {pid} en puerto {port}")
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True, check=False
                        )
        else:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True,
                text=True, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    print(f"Matando proceso {pid} en puerto {port}")
                    subprocess.run(["kill", "-9", pid], capture_output=True, check=False)
    except Exception as e:
        print(f"Warning: Error limpiando puerto {port}: {e}")

    time.sleep(2)


def start_vram_sampler(log_prefix: str):
    """Inicia el muestreador de VRAM con nvidia-smi (Linux)."""
    if platform.system() == "Windows":
        print("⚠️  Muestreo de VRAM no implementado en Windows")
        return None

    vram_log = f"{log_prefix}_vram.log"
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-lms", "500",
    ]
    with open(vram_log, "w") as f:
        return subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.DEVNULL, start_new_session=True
        )


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Espera a que el servidor esté listo."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/health", method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def send_request(port: int) -> Dict[str, Any]:
    """Envía la petición de prueba al servidor."""
    prompt_text = PROMPT.read_text(encoding="utf-8")
    request_data = {
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0,
        "n_predict": 128,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(request_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def get_max_vram(log_prefix: str) -> int:
    """Obtiene el máximo de VRAM del log del muestreador."""
    max_vram = 0
    try:
        with open(f"{log_prefix}_vram.log", "r") as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    max_vram = max(max_vram, int(line))
    except FileNotFoundError:
        pass
    return max_vram


def extract_metrics(log_prefix: str) -> Dict[str, str]:
    """Extrae métricas del log del servidor."""
    metrics = {}
    try:
        with open(f"{log_prefix}.log", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "prompt eval time" in line:
                    metrics["prompt_eval"] = line.strip()
                elif "eval time" in line and "prompt" not in line:
                    metrics["eval"] = line.strip()
                elif "total time" in line:
                    metrics["total"] = line.strip()
    except FileNotFoundError:
        pass
    return metrics


# ============================================================
# Pruebas
# ============================================================

def run_stock_server(config: Dict[str, Any], idx: int, total: int) -> bool:
    """Ejecuta un test de servidor stock lanzando llama-server directamente."""
    desc = config.get("desc", "Sin descripción")
    ncmoe = config["ncmoe"]
    load_mode = config["load_mode"]
    ctx = config["ctx"]
    threads = config.get("threads", 4)
    log_prefix = f"stock_server_ncmoe{ncmoe}_{load_mode}"

    print("\n" + "=" * 70)
    print(f"🧪 [{idx + 1}/{total}] {desc}")
    print("=" * 70)

    print("Limpiando servidores anteriores...")
    kill_server_on_port(PORT)
    if check_port_in_use(PORT):
        print(f"❌ ERROR: Puerto {PORT} todavía ocupado")
        return False

    cmd = [
        _binary("llama-server"),
        "-m", str(MODEL),
        "-ngl", "99", "-ncmoe", str(ncmoe), "-fa", "on",
        "-c", str(ctx), "-t", str(threads),
        "--load-mode", load_mode,
        "-np", "1", "--no-warmup",
        "--port", str(PORT), "--host", "0.0.0.0",
    ]
    print(f"💻 Comando: {' '.join(cmd)}")

    print("Iniciando muestreador de VRAM...")
    sampler = start_vram_sampler(log_prefix)

    log_file = f"{log_prefix}.log"
    print("Iniciando servidor...")
    with open(log_file, "w", encoding="utf-8") as f:
        server = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            start_new_session=(platform.system() != "Windows"),
        )

    print("Esperando a que el servidor esté listo...")
    ready = wait_for_server(PORT, timeout=300)
    if not ready:
        print("❌ ERROR: Timeout esperando al servidor")
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f.readlines()[-50:]:
                    print(line.rstrip())
        except FileNotFoundError:
            print("No se pudo abrir el log")
        if server.poll() is None:
            kill_process_tree(server.pid)
        if sampler:
            kill_process_tree(sampler.pid)
        return False

    print("✓ Servidor listo")

    ok = True
    print("\nEnviando petición...")
    try:
        response = send_request(PORT)
        print("✓ Petición completada")
        timings = response.get("timings")
        if timings:
            print("Timing:")
            print(f"  prompt: {timings['prompt_n']} tokens en "
                  f"{timings['prompt_ms']:.2f}ms "
                  f"({timings['prompt_per_second']:.2f} t/s)")
            print(f"  decode: {timings['predicted_n']} tokens en "
                  f"{timings['predicted_ms']:.2f}ms "
                  f"({timings['predicted_per_second']:.2f} t/s)")
    except Exception as e:
        print(f"❌ ERROR: Petición falló: {e}")
        ok = False

    print("\nLimpiando procesos...")
    kill_process_tree(server.pid)
    if sampler:
        kill_process_tree(sampler.pid)
    time.sleep(2)

    print("\n=== Resultados ===")
    print(f"VRAM máxima: {get_max_vram(log_prefix)} MiB")
    print("Métricas:")
    for value in extract_metrics(log_prefix).values():
        print(value)
    print("Logs:")
    print(f"  {log_prefix}.log")
    print(f"  {log_prefix}_vram.log")

    return ok


def run_stock_bench(config: Dict[str, Any], idx: int, total: int) -> bool:
    """Ejecuta un barrido de llama-bench stock."""
    desc = config.get("desc", "Sin descripción")
    ncmoe = config["ncmoe"]
    load_mode = config["load_mode"]
    log_prefix = f"stock_bench_ncmoe{ncmoe}_{load_mode}"

    print("\n" + "=" * 70)
    print(f"🧪 [{idx + 1}/{total}] {desc}")
    print("=" * 70)

    cmd = [
        _binary("llama-bench"),
        "-m", str(MODEL),
        "-ngl", "99", "-ncmoe", str(ncmoe), "-fa", "1",
        "-p", "0", "-n", "128", "-d", "0,4096",
        "-r", "3", "-t", "4",
        "-lm", load_mode, "-o", "json",
    ]
    print(f"💻 Comando: {' '.join(cmd)}")

    print("Iniciando muestreador de VRAM...")
    sampler = start_vram_sampler(log_prefix)

    try:
        with open(f"{log_prefix}.json", "w") as out, \
                open(f"{log_prefix}.stderr", "w") as err:
            result = subprocess.run(cmd, stdout=out, stderr=err)
        ok = result.returncode == 0
    except FileNotFoundError:
        print(f"❌ No se encontró el binario {cmd[0]}")
        ok = False
    finally:
        if sampler:
            kill_process_tree(sampler.pid)

    if ok:
        print(f"✓ Completado (VRAM máxima: {get_max_vram(log_prefix)} MiB)")
    else:
        print(f"❌ Falló (código {getattr(result, 'returncode', '?')})")
    print("Logs:")
    print(f"  {log_prefix}.json")
    print(f"  {log_prefix}_vram.log")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Suite stock llama.cpp")
    parser.add_argument(
        "--type",
        choices=["bench", "server", "all"],
        default="all",
        help="Tipo de tests a ejecutar"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Detener al primer fallo"
    )

    args = parser.parse_args()

    if not MODEL.exists():
        print(f"❌ ERROR: Modelo no encontrado en {MODEL}")
        sys.exit(1)
    if not PROMPT.exists():
        print(f"❌ ERROR: Prompt no encontrado en {PROMPT}")
        sys.exit(1)

    configs = []
    if args.type in ("bench", "all"):
        configs.extend([(c, "bench") for c in STOCK_BENCH_CONFIGS])
    if args.type in ("server", "all"):
        configs.extend([(c, "server") for c in STOCK_SERVER_CONFIGS])

    print(f"\n🚀 Suite stock llama.cpp: {len(configs)} configuraciones")

    results = []
    for idx, (config, test_type) in enumerate(configs):
        if test_type == "server":
            ok = run_stock_server(config, idx, len(configs))
        else:
            ok = run_stock_bench(config, idx, len(configs))

        results.append((config["desc"], ok))

        if not ok and args.stop_on_error:
            print("\n🛑 Deteniendo suite")
            break

    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)

    for desc, ok in results:
        status = "✅" if ok else "❌"
        print(f"{status} {desc}")

    successful = sum(1 for _, ok in results if ok)
    print(f"\nTotal: {successful}/{len(results)} exitosas")


if __name__ == "__main__":
    main()
