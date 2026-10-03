#!/usr/bin/env python3
"""
Script de benchmark para fork-moecache de llama.cpp
Compatible con Linux y Windows

Uso:
  python test-moe.py cache-only --slots N --ncmoe N --ctx N   # solo --moe-expert-cache-size
  python test-moe.py cache-full --slots N --ncmoe N --ctx N   # caché + flags experimentales
  python test-moe.py no-cache --slots N --ncmoe N --ctx N    # baseline sin caché
  python test-moe.py all --slots N --ncmoe N --ctx N         # los tres

Ejemplos:
  python test-moe.py all --slots 124 --ncmoe 20 --ctx 4096
  python test-moe.py cache-full --smoke-test                 # preflight (n=32)
"""

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Configuración
WORKDIR = Path("/workspace")
MODEL = WORKDIR / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
PROMPT = WORKDIR / "prompt2434.txt"
MOECACHE = WORKDIR / "fork-moecache" / "build" / "bin"
PORT = 8189
DEFAULT_SLOTS = 64
DEFAULT_NCMOE = 24
DEFAULT_CTX = 4096
DEFAULT_THREADS = 4  # i7-7700 4C/8T; coherente con el resto del arnés (-t 4)
DEFAULT_KV_TYPE = "q8_0"  # f16 gasta VRAM que iría a expertos; q8_0 por defecto

# Flags experimentales del fork que van JUNTOS (modo cache-full). El modo
# cache-only los deja fuera para aislar el efecto del --moe-expert-cache-size.
CACHE_FULL_FLAGS = [
    "--moe-early-router",
    "--ple-prefetch",
    "--backend-sampling",
    "--decode-overlap",
    "--decode-boundary-overlap",
    "--phase-aware-workspace",
    "--experimental-logs",
]

# Frases del log de llama-server que indican que la caché de expertos se montó.
# El smoke test aborta si ninguna aparece: la caché puede haberse ignorado
# en silencio (p. ej. por OOM) y la suite entera mediría ruido.
CACHE_EVIDENCE_KEYWORDS = (
    "expert cache",
    "moe cache",
    "cache size",
    "cache slots",
    "cache enabled",
    "cache loaded",
)


def kill_process_tree(pid: int):
    """Mata un proceso y todos sus hijos"""
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False
            )
        else:
            # Unix/Linux: enviar SIGTERM al grupo de procesos
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
    """Verifica si un puerto está en uso"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(('127.0.0.1', port))
        return result == 0


def kill_server_on_port(port: int):
    """Mata cualquier servidor en el puerto especificado"""
    try:
        if platform.system() == "Windows":
            # Windows: usar netstat para encontrar el PID
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=False
            )
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = int(parts[-1])
                        print(f"Matando proceso {pid} en puerto {port}")
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True,
                            check=False
                        )
        else:
            # Unix/Linux: usar lsof o fuser
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    print(f"Matando proceso {pid} en puerto {port}")
                    subprocess.run(
                        ["kill", "-9", pid],
                        capture_output=True,
                        check=False
                    )
    except Exception as e:
        print(f"Warning: Error limpiando puerto {port}: {e}")
    
    time.sleep(2)


def start_vram_sampler(log_prefix: str) -> subprocess.Popen:
    """Inicia el muestreador de VRAM"""
    vram_log = f"{log_prefix}_vram.log"
    
    if platform.system() == "Windows":
        # Windows: usar nvidia-smi en un bucle (simplificado)
        # Nota: En Windows necesitarías implementar un muestreador manual
        print("⚠️  Muestreo de VRAM no implementado en Windows, usando valor estático")
        return None
    else:
        # Unix/Linux
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-lms", "500"
        ]
        with open(vram_log, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        return proc


def build_server_command(
    test_mode: str,
    slots: int,
    ncmoe: int,
    ctx: int,
    threads: int = DEFAULT_THREADS,
    kv_type: str = DEFAULT_KV_TYPE,
    n_predict: int = 1024
) -> list:
    """Construye el comando del servidor"""
    llama_server = str(MOECACHE / "llama-server")
    if platform.system() == "Windows":
        llama_server += ".exe"

    cmd = [
        llama_server,
        "-m", str(MODEL),
        "-c", str(ctx), "-b", "2048", "-ub", "512", "-np", "1", "-n", str(n_predict),
        "--load-mode", "none", "--no-warmup", "-fa", "on",
        "-t", str(threads),
        "-ctk", kv_type, "-ctv", kv_type, "-kvo",
        "--fit", "off", "--jinja", "--reasoning", "off",
        "-ngl", "99", "--cache-ram", "0", "--spec-type", "none", "--log-colors", "off",
        "--lazy-mode", "on",
        "--n-cpu-moe", str(ncmoe),
        "--port", str(PORT), "--host", "0.0.0.0"
    ]

    # cache-only: solo el presupuesto de slots; cache-full: caché + flags juntos
    if test_mode in ("cache", "cache-only"):
        if test_mode == "cache":
            print("⚠️  'cache' es alias deprecado; usa 'cache-only' o 'cache-full'")
        cmd.extend(["--moe-expert-cache-size", str(slots)])
    elif test_mode == "cache-full":
        cmd.extend(["--moe-expert-cache-size", str(slots)])
        cmd.extend(CACHE_FULL_FLAGS)

    return cmd


def wait_for_server(timeout: int = 300) -> bool:
    """Espera a que el servidor esté listo"""
    import urllib.request
    import urllib.error
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/health",
                method='GET'
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        
        time.sleep(1)
    
    return False


def send_request(n_predict: int = 128) -> Dict[str, Any]:
    """Envía una petición al servidor"""
    import urllib.request
    
    prompt_text = PROMPT.read_text(encoding='utf-8')
    request_data = {
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0,
        "n_predict": n_predict,
        "stream": False
    }
    
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(request_data).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode('utf-8'))


def get_max_vram(log_prefix: str) -> int:
    """Obtiene el máximo de VRAM del log"""
    vram_log = f"{log_prefix}_vram.log"
    max_vram = 0
    
    try:
        with open(vram_log, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    max_vram = max(max_vram, int(line))
    except FileNotFoundError:
        pass
    
    return max_vram


def extract_metrics(log_prefix: str) -> Dict[str, str]:
    """Extrae métricas del log del servidor"""
    log_file = f"{log_prefix}.log"
    metrics = {}
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                if "prompt eval time" in line:
                    metrics['prompt_eval'] = line.strip()
                elif "eval time" in line and "prompt" not in line:
                    metrics['eval'] = line.strip()
                elif "total time" in line:
                    metrics['total'] = line.strip()
    except FileNotFoundError:
        pass
    
    return metrics


def run_test(
    test_mode: str,
    slots: int,
    ncmoe: int,
    ctx: int,
    threads: int = DEFAULT_THREADS,
    kv_type: str = DEFAULT_KV_TYPE,
    n_predict: int = 1024,
    smoke: bool = False,
    vram_limit_mib: int = 11500
):
    """Ejecuta una prueba. Con smoke=True hace una pasada corta (n=32) y
    valida que la caché entra en VRAM y se activa, abortando si no."""
    log_prefix = f"moe_{test_mode}_ncmoe{ncmoe}_c{ctx}"
    if smoke:
        n_predict = 32
        log_prefix = f"smoke_{log_prefix}"

    print("\n" + "=" * 50)
    print(f"Prueba: fork-moecache ({test_mode}{' [smoke]' if smoke else ''})")
    print(f"Config: ncmoe={ncmoe}, ctx={ctx}, slots={slots}, threads={threads}, kv={kv_type}")
    print("=" * 50)
    
    # Limpiar servidor previo
    print("Limpiando servidores anteriores...")
    kill_server_on_port(PORT)
    
    if check_port_in_use(PORT):
        print(f"❌ ERROR: Puerto {PORT} todavía ocupado")
        sys.exit(1)
    
    # Iniciar muestreador de VRAM
    print("Iniciando muestreador de VRAM...")
    sampler = start_vram_sampler(log_prefix)
    
    # Construir y ejecutar comando
    cmd = build_server_command(test_mode, slots, ncmoe, ctx, threads, kv_type, n_predict)
    log_file = f"{log_prefix}.log"
    
    print("Iniciando servidor...")
    with open(log_file, 'w', encoding='utf-8') as f:
        server = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=(platform.system() != "Windows")
        )
    
    # Esperar a que esté listo
    print("Esperando a que el servidor esté listo...")
    ready = wait_for_server(timeout=300)
    
    if not ready:
        print("❌ ERROR: Timeout esperando al servidor")
        print("\nÚltimas 50 líneas del log:")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line.rstrip())
        except FileNotFoundError:
            print("No se pudo abrir el log")
        
        if server.poll() is None:
            kill_process_tree(server.pid)
        if sampler:
            kill_process_tree(sampler.pid)
        sys.exit(1)
    
    print("✓ Servidor listo")
    
    # Estadísticas de caché
    if test_mode in ("cache", "cache-only", "cache-full"):
        print("\n=== Estadísticas de caché ===")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if any(kw in line.lower() for kw in ['moe', 'cache', 'expert', 'lru']):
                        print(line.rstrip())
                        if sum(1 for _ in open(log_file) if any(kw in _.lower() for kw in ['moe', 'cache'])) >= 10:
                            break
        except:
            pass
    
    # Enviar petición
    print("\nEnviando petición...")
    try:
        response = send_request(n_predict=n_predict)
        print("✓ Petición completada")
    except Exception as e:
        print(f"❌ ERROR: Petición falló: {e}")
        kill_process_tree(server.pid)
        if sampler:
            kill_process_tree(sampler.pid)
        sys.exit(1)
    
    # Limpieza
    print("\nLimpiando procesos...")
    kill_process_tree(server.pid)
    if sampler:
        kill_process_tree(sampler.pid)
    
    time.sleep(2)
    
    # Resultados
    print("\n" + "=" * 50)
    print(f"Resultados ({test_mode})")
    print("=" * 50)
    
    vram_max = get_max_vram(log_prefix)
    print(f"VRAM máxima: {vram_max} MiB")
    
    print("\nMétricas:")
    metrics = extract_metrics(log_prefix)
    for key, value in metrics.items():
        print(value)
    
    if 'timings' in response:
        timings = response['timings']
        print(f"\nTiming:")
        print(f"  prompt: {timings['prompt_n']} tokens en {timings['prompt_ms']:.2f}ms ({timings['prompt_per_second']:.2f} t/s)")
        print(f"  decode: {timings['predicted_n']} tokens en {timings['predicted_ms']:.2f}ms ({timings['predicted_per_second']:.2f} t/s)")
    
    if test_mode in ("cache", "cache-only", "cache-full"):
        print("\n=== Hit/Miss de caché ===")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                count = 0
                for line in f:
                    if any(kw in line.lower() for kw in ['hit', 'miss', 'cache']):
                        print(line.rstrip())
                        count += 1
                        if count >= 20:
                            break
        except:
            print("(sin datos)")
    
    print(f"\nLogs:")
    print(f"  {log_prefix}.log")
    print(f"  {log_prefix}_vram.log")
    # Validación smoke: la caché debe caber en VRAM y debe haberse activado.
    # Si el flag no se aplicó (p. ej. OOM silencioso), abortamos para no
    # correr la suite entera con una caché inexistente.
    if smoke and test_mode in ("cache", "cache-only", "cache-full"):
        if vram_max > vram_limit_mib:
            print(f"❌ VRAM pico {vram_max} MiB > límite {vram_limit_mib} MiB")
            print("   La caché puede haberse ignorado en silencio. Abortando.")
            sys.exit(1)
        print(f"✓ VRAM cabe: {vram_max} MiB ≤ {vram_limit_mib} MiB")

        cache_evidencia = False
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if any(kw in line.lower() for kw in CACHE_EVIDENCE_KEYWORDS):
                        cache_evidencia = True
                        break
        except FileNotFoundError:
            pass
        if not cache_evidencia:
            print("❌ Sin evidencia de caché en el log")
            print("   El flag puede haberse ignorado en silencio. Abortando.")
            sys.exit(1)
        print("✓ Evidencia de caché encontrada en el log")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark para fork-moecache de llama.cpp',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s cache-only --slots 124 --ncmoe 20 --ctx 4096   # Solo --moe-expert-cache-size
  %(prog)s cache-full --slots 124 --ncmoe 20 --ctx 4096   # Caché + flags experimentales juntos
  %(prog)s no-cache --slots 124 --ncmoe 20 --ctx 4096     # Baseline sin caché
  %(prog)s all --slots 124 --ncmoe 20 --ctx 4096          # Los tres
  %(prog)s cache-full --smoke-test --vram-limit 11500     # Preflight: valida VRAM y caché
        """
    )
    parser.add_argument(
        'mode',
        choices=['cache', 'cache-only', 'cache-full', 'no-cache', 'all'],
        help='Modo de prueba: cache-only (solo --moe-expert-cache-size), '
             'cache-full (caché + flags experimentales juntos), '
             'no-cache (baseline), all (los tres). cache = alias deprecado de cache-only'
    )
    parser.add_argument('--slots', type=int, default=DEFAULT_SLOTS,
                        help=f'Número de slots de caché (default: {DEFAULT_SLOTS})')
    parser.add_argument('--ncmoe', type=int, default=DEFAULT_NCMOE,
                        help=f'Número de capas MoE en CPU (default: {DEFAULT_NCMOE})')
    parser.add_argument('--ctx', type=int, default=DEFAULT_CTX,
                        help=f'Tamaño de contexto (default: {DEFAULT_CTX})')
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS,
                        help=f'Hilos CPU (default: {DEFAULT_THREADS}; i7-7700 4C/8T)')
    parser.add_argument('--kv-type', choices=['q8_0', 'f16'], default=DEFAULT_KV_TYPE,
                        help=f'KV cache quant (default: {DEFAULT_KV_TYPE}; f16 gasta VRAM que iría a expertos)')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Pasada corta (n=32): valida que la caché entra en VRAM y se activa; '
                             'aborta (exit 1) si no. Pensado para preflight de la suite.')
    parser.add_argument('--vram-limit', type=int, default=11500,
                        help=f'Límite VRAM en MiB para smoke test (default: 11500)')
    args = parser.parse_args()

    # Verificaciones
    if not MODEL.exists():
        print(f"❌ ERROR: Modelo no encontrado en {MODEL}")
        sys.exit(1)

    if not PROMPT.exists():
        print(f"❌ ERROR: Prompt no encontrado en {PROMPT}")
        sys.exit(1)

    common = dict(
        threads=args.threads,
        kv_type=args.kv_type,
        smoke=args.smoke_test,
        vram_limit_mib=args.vram_limit,
    )

    # Preflight (--smoke-test): solo valida, no genera la comparación
    if args.smoke_test:
        if args.mode in ('cache', 'cache-only', 'cache-full'):
            run_test(args.mode, args.slots, args.ncmoe, args.ctx, **common)
        elif args.mode == 'all':
            for mode in ('cache-only', 'cache-full'):
                run_test(mode, args.slots, args.ncmoe, args.ctx, **common)
        # 'no-cache' en smoke: nada que validar, se acepta sin corrida
        return 0

    # Suite completa
    if args.mode in ('cache', 'cache-only', 'no-cache'):
        run_test(args.mode, args.slots, args.ncmoe, args.ctx, **common)

    elif args.mode == 'cache-full':
        run_test('cache-full', args.slots, args.ncmoe, args.ctx, **common)

    elif args.mode == 'all':
        run_test('no-cache', args.slots, args.ncmoe, args.ctx, **common)
        run_test('cache-only', args.slots, args.ncmoe, args.ctx, **common)
        run_test('cache-full', args.slots, args.ncmoe, args.ctx, **common)

        print("\n" + "=" * 50)
        print("Comparación final")
        print("=" * 50)

        print("\nSin caché:")
        metrics = extract_metrics(f"moe_no-cache_ncmoe{args.ncmoe}_c{args.ctx}")
        for value in metrics.values():
            print(value)

        print("\nSolo caché:")
        metrics = extract_metrics(f"moe_cache-only_ncmoe{args.ncmoe}_c{args.ctx}")
        for value in metrics.values():
            print(value)

        print("\nCaché + flags:")
        metrics = extract_metrics(f"moe_cache-full_ncmoe{args.ncmoe}_c{args.ctx}")
        for value in metrics.values():
            print(value)


if __name__ == '__main__':
    main()