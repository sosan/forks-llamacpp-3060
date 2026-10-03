#!/usr/bin/env python3

"""Calculadora MoE: cuántos expertos caben en VRAM.

Unidades: todo en MiB salvo que se indique.

V_libre = V_usable - V_base - V_compute - margen

slots = floor(V_libre / (N_capas * S_exp))
        # expertos por capa que pueden mantenerse en caché estática

capas = floor(V_libre / (E * S_exp))
        # capas MoE completas que pueden residir en GPU

fraccion = slots / E

"""

import math


def s_exp_por_medicion(v1, v2, ng1, ng2, e):
    return abs(v1 - v2) / abs(ng1 - ng2) / e


def calcula(v_usable, v_base, v_compute, margen,
            n_capas, e, s_exp):

    v_libre = v_usable - v_base - v_compute - margen

    s_capa = e * s_exp
    por_slot = n_capas * s_exp

    slots = (
        max(0, math.floor(v_libre / por_slot))
        if por_slot > 0 else 0
    )

    capas = (
        max(0, math.floor(v_libre / s_capa))
        if s_capa > 0 else 0
    )

    frac = slots / e if e else 0.0

    veredicto = (
        "SI compensa (>=15-20%)"
        if frac >= 0.15
        else "NO compensa (<15%)"
    )

    if 0.15 <= frac < 0.20:
        veredicto += " (zona limite)"

    return {
        "v_libre": v_libre,
        "s_capa": s_capa,
        "slots": slots,
        "capas": capas,
        "frac": frac,
        "veredicto": veredicto,
    }


def pide(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return type(default)(s) if s else default


def main():

    print(
        "Deja vacio para aceptar el valor entre corchetes "
        "(ejemplo 3060)."
    )

    v_usable = pide(
        "VRAM usable real de tu GPU en MiB",
        11909.0
    )

    n_capas = pide(
        "Numero de capas MoE que tiene el modelo",
        40
    )

    e = pide(
        "Numero de expertos que tiene cada capa",
        256
    )

    v_compute = pide(
        "VRAM extra que reservan los buffers de computo en MiB",
        0.0
    )

    margen = pide(
        "Margen de seguridad que quieres dejar libre en MiB",
        0.0
    )

    print(
        "\nPara medir el pico de VRAM con 0 capas MoE en GPU, primero"
    )

    print(
        "carga 0 capas MoE en GPU con este comando:"
    )

    print(
        "  nvidia-smi --query-gpu=memory.used "
        "--format=csv -lms 500 > vram_99.log &"
    )

    print(
        "  llama-bench -m modelo.gguf -ngl 99 "
        "-ncmoe 99 -fa 1 -p 0 -n 128 "
        "-d 0,4096 -r 3 -t 4 -o json"
    )

    v_base = pide(
        "Resultado del pico de VRAM con 0 capas MoE "
        "en GPU (-ncmoe 99) en MiB",
        2765.0
    )

    print(
        "\nPara el segundo pico, aumenta el numero de capas MoE en GPU."
    )

    print(
        "Puedes usar una sola capa o tantas como quepan sin OOM."
    )

    print(
        "Cuantas mas capas difieran entre las dos corridas, "
        "mayor sera la senal frente al ruido del muestreo."
    )

    print(
        "  nvidia-smi --query-gpu=memory.used "
        "--format=csv -lms 500 > vram_N.log &"
    )

    print(
        f"  llama-bench -m modelo.gguf -ngl 99 "
        f"-ncmoe {n_capas - 1} -fa 1 -p 0 "
        f"-n 128 -d 0,4096 -r 3 -t 4 -o json"
    )

    ng2 = pide(
        "Capas MoE en GPU durante la segunda corrida",
        16
    )

    if ng2 == 0:
        print(
            "La segunda corrida tambien tiene 0 capas: "
            "sin diferencia no hay nada que medir."
        )
        return

    v2 = pide(
        f"Resultado del pico de VRAM con {ng2} capa(s) "
        f"MoE en GPU en MiB",
        10259.0
    )

    s_exp = s_exp_por_medicion(
        v_base,
        v2,
        0,
        ng2,
        e
    )

    s_capa = s_exp * e

    if not 50 <= s_capa <= 2000:
        print(
            f"Ojo: una capa de {s_capa:.0f} MiB suena rara. "
            "Revisa que ambos picos sean del mismo "
            "contexto/build y que las capas sean correctas."
        )

    print(
        f"Una capa completa ocupa = {s_capa:.0f} MiB, "
        f"un experto = {s_exp:.2f} MiB"
    )

    r = calcula(
        v_usable,
        v_base,
        v_compute,
        margen,
        n_capas,
        e,
        s_exp
    )

    print(
        f"\nVRAM libre para expertos = "
        f"{r['v_libre']:.0f} MiB"
    )

    print(
        f"Capas completas que caben en GPU: "
        f"{r['capas']}"
    )

    print(
        f"Expertos por capa (cache estatica): "
        f"{r['slots']} "
        f"({r['frac']:.1%} de {e}) "
        f"-> {r['veredicto']}"
    )


if __name__ == "__main__":
    main()