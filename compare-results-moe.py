#!/usr/bin/env python3
"""
Genera tabla comparativa de todos los tests ejecutados
Solo muestra resultados en pantalla
"""

import re
from pathlib import Path
from typing import Dict

def extract_results(log_prefix: str) -> Dict:
    """Extrae resultados de un log"""
    results = {
        'config': log_prefix,
        'vram_max': 0,
        'prompt_tps': 0,
        'decode_tps': 0,
        'total_time': 0
    }
    
    # VRAM
    vram_file = Path(f"{log_prefix}_vram.log")
    if vram_file.exists():
        with open(vram_file) as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    results['vram_max'] = max(results['vram_max'], int(line))
    
    # Métricas del servidor
    log_file = Path(f"{log_prefix}.log")
    if log_file.exists():
        with open(log_file, encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Buscar métricas con patrones más flexibles
            prompt_match = re.search(r'prompt eval time.*?(\d+\.\d+) tokens per second', content)
            if prompt_match:
                results['prompt_tps'] = float(prompt_match.group(1))
            
            decode_match = re.search(r'\beval time\b.*?(\d+\.\d+) tokens per second', content)
            if decode_match:
                results['decode_tps'] = float(decode_match.group(1))
            
            total_match = re.search(r'total time.*?(\d+\.\d+) ms', content)
            if total_match:
                results['total_time'] = float(total_match.group(1))
    
    return results

def main():
    # Buscar todos los logs
    logs = sorted(Path('.').glob('moe_*_ncmoe*_c*.log'))
    
    results = []
    for log in logs:
        prefix = log.stem
        if not prefix.endswith('_vram'):
            result = extract_results(prefix)
            if result['prompt_tps'] > 0:  # Solo incluir tests completados
                results.append(result)
    
    if not results:
        print("\n⚠️  No se encontraron resultados de tests completados")
        print("Ejecuta primero algunos tests con: python test-moe.py [mode] [slots] [ncmoe] [ctx]")
        return
    
    # Ordenar por decode throughput (mejor a peor)
    results.sort(key=lambda x: x['decode_tps'], reverse=True)
    
    # Imprimir tabla
    print("\n" + "=" * 100)
    print("TABLA COMPARATIVA DE RESULTADOS")
    print("=" * 100)
    print(f"{'Configuración':<45} {'VRAM (MiB)':<12} {'Prompt (t/s)':<15} {'Decode (t/s)':<15} {'Total (ms)':<12}")
    print("-" * 100)
    
    for r in results:
        # Resaltar el mejor resultado
        marker = " 👑" if r == results[0] else ""
        print(f"{r['config']:<45} {r['vram_max']:<12} {r['prompt_tps']:<15.2f} {r['decode_tps']:<15.2f} {r['total_time']:<12.0f}{marker}")
    
    print("=" * 100)
    
    # Estadísticas resumidas
    if results:
        best = results[0]
        print(f"\n🏆 Mejor configuración: {best['config']}")
        print(f"   Decode: {best['decode_tps']:.2f} t/s | VRAM: {best['vram_max']} MiB")
        
        if len(results) > 1:
            worst = results[-1]
            improvement = ((best['decode_tps'] - worst['decode_tps']) / worst['decode_tps']) * 100
            print(f"\n📊 Mejora vs peor configuración: +{improvement:.1f}%")

if __name__ == "__main__":
    main()