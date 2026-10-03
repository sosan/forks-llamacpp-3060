#!/usr/bin/env python3
"""Calculadora MoE: cuántos expertos caben en VRAM (unidades: MiB).

Modo interactivo (Enter acepta el valor entre corchetes):
    python3 calc.py

Modo parametrizado (sin preguntas):
    # RTX 3060 a 4K, derivando S_exp de dos picos (0 y 16 capas)
    python3 calc.py --v-usable 11909 --v-base 2765 --v2 10259 --ng2 16

    # GTX 1650 a 4K, dando el tamaño de capa directamente
    python3 calc.py --v-usable 3714 --v-base 2701 --s-capa 470 --margen 100

    # S_exp ya conocido
    python3 calc.py --v-usable 11909 --v-base 2765 --s-exp 1.83
"""

import argparse
import math
import sys


def s_exp_por_medicion(v1, v2, ng1, ng2, e):
    return abs(v1 - v2) / abs(ng1 - ng2) / e


def calcula(v_usable, v_base, v_compute, margen, n_capas, e, s_exp):
    v_libre = v_usable - v_base - v_compute - margen
    s_capa = e * s_exp
    por_slot = n_capas * s_exp

    slots = max(0, math.floor(v_libre / por_slot)) if por_slot > 0 else 0
    capas = max(0, math.floor(v_libre / s_capa)) if s_capa > 0 else 0
    frac = slots / e if e else 0.0

    veredicto = "SI compensa (>=15-20%)" if frac >= 0.15 else "NO compensa (<15%)"
    if 0.15 <= frac < 0.20:
        veredicto += " (zona limite)"

    return {"v_libre": v_libre, "s_capa": s_capa, "slots": slots,
            "capas": capas, "frac": frac, "veredicto": veredicto}


def pide(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return type(default)(s) if s else default


def aviso_s_capa(s_capa):
    if not 50 <= s_capa <= 2000:
        print(f"Ojo: una capa de {s_capa:.0f} MiB suena rara. Revisa que ambos "
              "picos sean del mismo contexto/build y que las capas sean correctas.")


def resuelve_s_exp(args):
    """Devuelve S_exp a partir de --s-exp, --s-capa o dos mediciones."""
    if args.s_exp is not None:
        return args.s_exp
    if args.s_capa is not None:
        return args.s_capa / args.e
    if args.v2 is not None and args.ng2 is not None:
        if args.ng2 == 0:
            raise SystemExit("La segunda corrida tambien tiene 0 capas: "
                             "sin diferencia no hay nada que medir.")
        return s_exp_por_medicion(args.v_base, args.v2, 0, args.ng2, args.e)
    raise SystemExit("Indica S_exp con --s-exp, --s-capa o una segunda medicion "
                     "(--v2 y --ng2).")


def reporta(r, e):
    print(f"Una capa completa ocupa {r['s_capa']:.0f} MiB, "
          f"un experto {r['s_capa'] / e:.2f} MiB")
    print(f"\nVRAM libre para expertos = {r['v_libre']:.0f} MiB")
    print(f"Capas completas que caben en GPU: {r['capas']}")
    print(f"Expertos por capa (cache estatica): {r['slots']} "
          f"({r['frac']:.1%} de {e}) -> {r['veredicto']}")


def modo_interactivo():
    print("Deja vacio para aceptar el valor entre corchetes (ejemplo 3060).")

    v_usable = pide("VRAM usable real de tu GPU en MiB", 11909.0)
    n_capas = pide("Numero de capas MoE que tiene el modelo", 40)
    e = pide("Numero de expertos que tiene cada capa", 256)
    v_compute = pide("VRAM extra de los buffers de computo en MiB", 0.0)
    margen = pide("Margen de seguridad que quieres dejar libre en MiB", 0.0)

    print("\nPrimera medicion: 0 capas MoE en GPU.")
    print("  nvidia-smi --query-gpu=memory.used --format=csv -lms 500 > vram_99.log &")
    print("  llama-bench -m modelo.gguf -ngl 99 -ncmoe 99 -fa 1 -p 0 -n 128 "
          "-d 0,4096 -r 3 -t 4 -o json")
    v_base = pide("Pico de VRAM con 0 capas MoE en GPU (-ncmoe 99) en MiB", 2765.0)

    print("\nSegunda medicion: mas capas MoE en GPU (las que quepan sin OOM).")
    print("Cuantas mas capas difieran entre las dos, mas senal frente al ruido.")
    ng2 = pide("Capas MoE en GPU durante la segunda corrida", 16)
    if ng2 == 0:
        print("La segunda corrida tambien tiene 0 capas: sin diferencia no hay nada que medir.")
        return

    print("  nvidia-smi --query-gpu=memory.used --format=csv -lms 500 > vram_N.log &")
    print(f"  llama-bench -m modelo.gguf -ngl 99 -ncmoe {n_capas - ng2} -fa 1 "
          f"-p 0 -n 128 -d 0,4096 -r 3 -t 4 -o json")
    v2 = pide(f"Pico de VRAM con {ng2} capa(s) MoE en GPU en MiB", 10259.0)

    s_exp = s_exp_por_medicion(v_base, v2, 0, ng2, e)
    aviso_s_capa(s_exp * e)

    r = calcula(v_usable, v_base, v_compute, margen, n_capas, e, s_exp)
    reporta(r, e)


def modo_parametrizado(args):
    if args.v_usable is None or args.v_base is None:
        raise SystemExit("Faltan --v-usable y/o --v-base. Usa --help o ejecuta "
                         "sin argumentos para el modo interactivo.")

    s_exp = resuelve_s_exp(args)
    aviso_s_capa(s_exp * args.e)

    r = calcula(args.v_usable, args.v_base, args.v_compute, args.margen,
                args.n_capas, args.e, s_exp)
    reporta(r, args.e)


def build_parser():
    p = argparse.ArgumentParser(
        description="Calculadora MoE: cuantos expertos caben en VRAM (MiB).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s --v-usable 11909 --v-base 2765 --v2 10259 --ng2 16
  %(prog)s --v-usable 3714 --v-base 2701 --s-capa 470 --margen 100
  %(prog)s --v-usable 11909 --v-base 2765 --s-exp 1.83
        """,
    )
    p.add_argument("--v-usable", type=float, default=None,
                   help="VRAM usable real de la GPU en MiB")
    p.add_argument("--v-base", type=float, default=None,
                   help="VRAM con 0 capas MoE en GPU en MiB (-ncmoe 99)")
    p.add_argument("--v-compute", type=float, default=0.0,
                   help="VRAM extra de los buffers de computo en MiB (default: 0)")
    p.add_argument("--margen", type=float, default=0.0,
                   help="Margen de seguridad a dejar libre en MiB (default: 0)")
    p.add_argument("--n-capas", type=int, default=40,
                   help="Numero de capas MoE del modelo (default: 40)")
    p.add_argument("--e", type=int, default=256,
                   help="Numero de expertos por capa (default: 256)")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--s-exp", type=float, default=None,
                   help="Tamano de un experto en MiB (si ya lo conoces)")
    g.add_argument("--s-capa", type=float, default=None,
                   help="Tamano de una capa MoE en MiB (alternativa a --s-exp)")
    p.add_argument("--v2", type=float, default=None,
                   help="Pico de VRAM de la segunda corrida en MiB")
    p.add_argument("--ng2", type=int, default=None,
                   help="Capas MoE en GPU durante la segunda corrida")

    p.add_argument("--interactive", action="store_true",
                   help="Forzar el modo interactivo")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.interactive or args.v_usable is None:
        modo_interactivo()
    else:
        modo_parametrizado(args)


if __name__ == "__main__":
    main()
