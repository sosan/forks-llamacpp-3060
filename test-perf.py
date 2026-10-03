#!/usr/bin/env python3
"""
Script de benchmark para fork-perf de llama.cpp
Compatible con Linux y Windows

Uso:
  python test-perf.py cache [--slots N] [--ncmoe N] [--ctx N] [--mtp]
  python test-perf.py no-cache [--slots N] [--ncmoe N] [--ctx N] [--mtp]
  python test-perf.py both [--slots N] [--ncmoe N] [--ctx N] [--mtp]

Ejemplos:
  python test-perf.py cache                           # Con caché, sin MTP (defaults)
  python test-perf.py both --mtp                      # Comparar con/sin caché, con MTP
  python test-perf.py cache --slots 32 --ncmoe 24     # 32 slots, ncmoe 24
  python test-perf.py both --slots 64 --ncmoe 24 --ctx 4096 --mtp
"""

import argparse
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
# Configuración
# ============================================================
WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))
MODEL = WORKDIR / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
PROMPT = WORKDIR / "prompt2434.txt"
PERF = WORKDIR / "fork-perf" / "build" / "bin"
PROFILE = WORKDIR / "traces" / "qwen35b-merged.csv"

PORT = 8189
DEFAULT_SLOTS = 64
DEFAULT_NCMOE = 24
DEFAULT_CTX = 4096
HEALTH_TIMEOUT = 300
REQUEST_TIMEOUT = 300
N_PREDICT = 128


# ============================================================
# Utilidades cross-platform
# ============================================================

def is_windows() -> bool:
    return platform.system() == "Windows"


def server_binary() -> str:
    """Devuelve la ruta al binario de llama-server según el SO"""
    binary = PERF / "llama-server"
    if is_windows():
        binary = binary.with_suffix(".exe")
    return str(binary)


def kill_process_tree(pid: int) -> None:
    """Mata un proceso y todos sus hijos de forma cross-platform"""
    if pid is None:
        return
    try:
        if is_windows():
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                time.sleep(1)
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
    except Exception as e:
        print(f"⚠️  Warning: no se pudo matar proceso {pid}: {e}")


def kill_server_on_port(port: int) -> None:
    """Mata cualquier proceso que esté escuchando en el puerto dado"""
    try:
        if is_windows():
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, check=False
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pid = int(parts[-1])
                        print(f"  Matando PID {pid} en puerto {port}")
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True,
                            check=False,
                        )
        else:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                for pid_str in result.stdout.strip().splitlines():
                    print(f"  Matando PID {pid_str} en puerto {port}")
                    subprocess.run(
                        ["kill", "-9", pid_str], capture_output=True, check=False
                    )
    except FileNotFoundError:
        # lsof/netstat no disponibles
        pass
    except Exception as e:
        print(f"⚠️  Warning limpiando puerto {port}: {e}")

    time.sleep(2)


def check_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


# ============================================================
# VRAM sampler
# ============================================================

def start_vram_sampler(log_prefix: str) -> Optional[subprocess.Popen]:
    """Inicia el muestreador de VRAM (solo Linux con nvidia-smi)"""
    vram_log = f"{log_prefix}_vram.log"

    if is_windows():
        # En Windows nvidia-smi no soporta -lms de la misma forma;
        # implementamos un muestreador en bucle
        print("  Muestreador de VRAM en modo Windows (bucle)")
        script = f"""
import subprocess, time
with open(r"{vram_log}", "w") as f:
    while True:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                f.write(r.stdout.strip() + "\\n")
                f.flush()
        except Exception:
            pass
        time.sleep(0.5)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if is_windows() else 0,
        )
        return proc

    # Linux / macOS
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-lms", "500",
    ]
    with open(vram_log, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return proc


# ============================================================
# Construcción del comando
# ============================================================

def build_server_command(
    test_mode: str,
    slots: int,
    ncmoe: int,
    ctx: int,
    use_mtp: bool,
) -> List[str]:
    """Construye el comando del servidor fork-perf"""
    cmd = [
        server_binary(),
        "-m", str(MODEL),
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "-ngl", "99",
        "-ncmoe", str(ncmoe),
        "-fa", "on",
        "-c", str(ctx),
        "-t", "4",
        "-np", "1",
        "--load-mode", "none",
        "--no-warmup",
        "--temp", "0",
        "--top-k", "1",
        "--top-p", "1",
    ]

    # MTP opcional
    if use_mtp:
        cmd += [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "2",
        ]

    # Caché solo en modo cache
    if test_mode == "cache":
        cmd += [
            "--moe-cache-profile", str(PROFILE),
            "--moe-cache-slots", str(slots),
        ]

    return cmd


# ============================================================
# Health check y petición
# ============================================================

def wait_for_server(timeout: int = HEALTH_TIMEOUT) -> bool:
    """Espera a que /health devuelva HTTP 200"""
    start = time.time()
    last_progress = 0
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/health", method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError,
                ConnectionError, OSError, TimeoutError):
            pass

        elapsed = int(time.time() - start)
        if elapsed >= last_progress + 10:
            print(f"  ... esperando ({elapsed}s)")
            last_progress = elapsed
        time.sleep(1)

    return False


def send_request() -> Dict[str, Any]:
    """Envía una petición de chat completions"""
    prompt_text = PROMPT.read_text(encoding="utf-8")
    payload = {
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0,
        "n_predict": N_PREDICT,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ============================================================
# Análisis de resultados
# ============================================================

def get_max_vram(log_prefix: str) -> int:
    vram_log = f"{log_prefix}_vram.log"
    max_vram = 0
    try:
        with open(vram_log) as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    max_vram = max(max_vram, int(line))
    except FileNotFoundError:
        pass
    return max_vram


def print_server_metrics(log_prefix: str) -> None:
    log_file = f"{log_prefix}.log"
    try:
        with open(log_file, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        for line in lines:
            if any(kw in line for kw in ("prompt eval time", "eval time", "total time")):
                print(line.rstrip())
    except FileNotFoundError:
        print("  (log no encontrado)")


def print_cache_stats(log_prefix: str) -> None:
    log_file = f"{log_prefix}.log"
    try:
        with open(log_file, encoding="utf-8", errors="ignore") as f:
            count = 0
            for line in f:
                low = line.lower()
                if any(kw in low for kw in ("moe", "cache", "slots", "profile", "expert")):
                    print(line.rstrip())
                    count += 1
                    if count >= 10:
                        break
    except FileNotFoundError:
        print("  (log no encontrado)")


def print_hit_miss(log_prefix: str) -> None:
    log_file = f"{log_prefix}.log"
    try:
        with open(log_file, encoding="utf-8", errors="ignore") as f:
            count = 0
            for line in f:
                low = line.lower()
                if any(kw in low for kw in ("hit", "miss", "draft")):
                    print(line.rstrip())
                    count += 1
                    if count >= 20:
                        break
    except FileNotFoundError:
        print("  (sin datos)")


def print_timings_json(response: Dict[str, Any]) -> None:
    t = response.get("timings")
    if not t:
        return
    print("\nTiming (JSON):")
    print(f"  prompt: {t.get('prompt_n')} tokens en "
          f"{t.get('prompt_ms', 0):.2f}ms ({t.get('prompt_per_second', 0):.2f} t/s)")
    print(f"  decode: {t.get('predicted_n')} tokens en "
          f"{t.get('predicted_ms', 0):.2f}ms ({t.get('predicted_per_second', 0):.2f} t/s)")
    if "draft_n" in t:
        accepted = t.get("draft_n_accepted", 0)
        generated = t.get("draft_n", 0)
        rate = (accepted / generated * 100) if generated else 0
        print(f"  MTP drafts: {accepted}/{generated} aceptados ({rate:.1f}%)")


# ============================================================
# Ejecución de una prueba
# ============================================================

def run_test(
    test_mode: str,
    slots: int,
    ncmoe: int,
    ctx: int,
    use_mtp: bool,
) -> bool:
    mtp_tag = "_mtp" if use_mtp else ""
    log_prefix = f"perf_{test_mode}_ncmoe{ncmoe}_c{ctx}{mtp_tag}"

    print("\n" + "=" * 60)
    print(f"Prueba: fork-perf ({test_mode})")
    print(f"Config: ncmoe={ncmoe}, ctx={ctx}, slots={slots}, MTP={use_mtp}")
    print("=" * 60)

    # Limpiar servidor previo
    print("Limpiando servidores anteriores...")
    kill_server_on_port(PORT)

    if check_port_in_use(PORT):
        print(f"❌ ERROR: Puerto {PORT} todavía ocupado")
        return False

    # Muestreador de VRAM
    print("Iniciando muestreador de VRAM...")
    sampler = start_vram_sampler(log_prefix)

    # Servidor
    cmd = build_server_command(test_mode, slots, ncmoe, ctx, use_mtp)
    log_file = f"{log_prefix}.log"

    print("Iniciando servidor...")
    popen_kwargs: Dict[str, Any] = {
        "stdout": open(log_file, "w", encoding="utf-8"),
        "stderr": subprocess.STDOUT,
    }
    if not is_windows():
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    server = subprocess.Popen(cmd, **popen_kwargs)

    # Esperar health check
    print("Esperando a que el servidor esté listo...")
    ready = wait_for_server(HEALTH_TIMEOUT)

    if server.poll() is not None:
        print("❌ ERROR: El servidor murió durante el arranque")
        print("\nÚltimas 50 líneas del log:")
        try:
            with open(log_file, encoding="utf-8", errors="ignore") as f:
                for line in f.readlines()[-50:]:
                    print(line.rstrip())
        except FileNotFoundError:
            pass
        kill_process_tree(sampler.pid if sampler else None)
        return False

    if not ready:
        print(f"❌ ERROR: Timeout esperando al servidor ({HEALTH_TIMEOUT}s)")
        kill_process_tree(server.pid)
        kill_process_tree(sampler.pid if sampler else None)
        return False

    elapsed = int(time.time())
    print(f"✓ Servidor listo")

    # Estadísticas de caché
    if test_mode == "cache":
        print("\n=== Estadísticas de caché ===")
        print_cache_stats(log_prefix)

    # Petición
    print("\nEnviando petición...")
    try:
        response = send_request()
    except Exception as e:
        print(f"❌ ERROR: Petición falló: {e}")
        kill_process_tree(server.pid)
        kill_process_tree(sampler.pid if sampler else None)
        return False

    print("✓ Petición completada")

    # Limpieza
    print("Limpiando procesos...")
    kill_process_tree(server.pid)
    kill_process_tree(sampler.pid if sampler else None)
    time.sleep(2)

    # Resultados
    print("\n" + "=" * 60)
    print(f"Resultados ({test_mode})")
    print("=" * 60)

    vram_max = get_max_vram(log_prefix)
    print(f"VRAM máxima: {vram_max} MiB")

    print("\nMétricas del servidor:")
    print_server_metrics(log_prefix)

    print_timings_json(response)

    if test_mode == "cache":
        print("\n=== Hit/Miss de caché ===")
        print_hit_miss(log_prefix)

    print(f"\nLogs:")
    print(f"  {log_prefix}.log")
    print(f"  {log_prefix}_vram.log")

    return True


# ============================================================
# Comparación final
# ============================================================

def print_comparison(ncmoe: int, ctx: int, use_mtp: bool) -> None:
    mtp_tag = "_mtp" if use_mtp else ""
    print("\n" + "=" * 60)
    print("Comparación final")
    print("=" * 60)

    for mode in ("no-cache", "cache"):
        prefix = f"perf_{mode}_ncmoe{ncmoe}_c{ctx}{mtp_tag}"
        print(f"\n{'Sin caché' if mode == 'no-cache' else 'Con caché'}:")
        print_server_metrics(prefix)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark para fork-perf de llama.cpp (cross-platform)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s cache                                  # Con caché, sin MTP (defaults)
  %(prog)s no-cache                               # Sin caché, sin MTP
  %(prog)s both --mtp                             # Comparar con/sin caché, con MTP
  %(prog)s cache --slots 32 --ncmoe 24            # 32 slots, ncmoe 24
  %(prog)s both --slots 64 --ncmoe 24 --ctx 4096 --mtp
        """,
    )

    parser.add_argument(
        "mode",
        choices=["cache", "no-cache", "both"],
        help="Modo de prueba",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=DEFAULT_SLOTS,
        help=f"Número de slots de caché (default: {DEFAULT_SLOTS})",
    )
    parser.add_argument(
        "--ncmoe",
        type=int,
        default=DEFAULT_NCMOE,
        help=f"Capas MoE en CPU (default: {DEFAULT_NCMOE})",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=DEFAULT_CTX,
        help=f"Tamaño de contexto (default: {DEFAULT_CTX})",
    )
    parser.add_argument(
        "--mtp",
        action="store_true",
        help="Activar Multi-Token Prediction (--spec-type draft-mtp)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help=f"Directorio de trabajo (default: {WORKDIR})",
    )

    return parser.parse_args()


def main() -> None:
    global WORKDIR, MODEL, PROMPT, PERF, PROFILE

    args = parse_args()

    # Permitir sobreescribir WORKDIR por CLI
    if args.workdir is not None:
        WORKDIR = args.workdir
        MODEL = WORKDIR / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
        PROMPT = WORKDIR / "prompt2434.txt"
        PERF = WORKDIR / "fork-perf" / "build" / "bin"
        PROFILE = WORKDIR / "traces" / "qwen35b-merged.csv"

    # Verificaciones previas
    if not MODEL.exists():
        print(f"❌ ERROR: Modelo no encontrado en {MODEL}")
        sys.exit(1)

    if not PROMPT.exists():
        print(f"❌ ERROR: Prompt no encontrado en {PROMPT}")
        sys.exit(1)

    if not Path(server_binary()).exists():
        print(f"❌ ERROR: Binario no encontrado en {server_binary()}")
        sys.exit(1)

    if args.mode == "cache" and not PROFILE.exists():
        print(f"❌ ERROR: Perfil de trazas no encontrado en {PROFILE}")
        print("   Genera primero las trazas con llama-moe-trace")
        sys.exit(1)

    print(f"🚀 fork-perf benchmark")
    print(f"   SO: {platform.system()} | Puerto: {PORT}")
    print(f"   Modelo: {MODEL.name}")
    print(f"   MTP: {'activado' if args.mtp else 'desactivado'}")

    # Ejecutar pruebas
    if args.mode in ("cache", "no-cache"):
        ok = run_test(args.mode, args.slots, args.ncmoe, args.ctx, args.mtp)
        sys.exit(0 if ok else 1)

    elif args.mode == "both":
        ok1 = run_test("no-cache", args.slots, args.ncmoe, args.ctx, args.mtp)
        ok2 = run_test("cache", args.slots, args.ncmoe, args.ctx, args.mtp)

        if ok1 and ok2:
            print_comparison(args.ncmoe, args.ctx, args.mtp)

        sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()