#!/usr/bin/env python3
"""
test_cache_reuse.py — A aposta da config atual: ctx 131072 só é viável se
ctx-checkpoints + cache-ram grande evitarem o re-prefill total a cada turno.

Modelo híbrido SSM (qwen35) costuma "forçar full prompt re-processing". Este
teste mede se um SEGUNDO turno, que só ESTENDE o prefixo (padrão agêntico de
anexar tool output), reaproveita o KV ou re-processa tudo.

Sinal-chave: timings.prompt_n na 2ª request.
  - ~ (tokens novos)  → REUSO funciona  ✅
  - ~ (prompt inteiro) → full reprocess (os ~190s voltam) ❌

Uso:
    python test_cache_reuse.py            # base 50k tokens
    python test_cache_reuse.py --n 120000 # perto do teto real
"""

import argparse
import json
import sys
import time
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def post(base, path, body, timeout=900):
    data = json.dumps(body).encode("utf-8")
    r = urllib.request.Request(base + path, data=data, method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def tokenize(base, text):
    return post(base, "/tokenize", {"content": text})["tokens"]


def build_tokens(base, n):
    filler = ("The quick brown fox jumps over the lazy dog near the riverbank "
              "while the sun slowly sets behind the distant mountains. ")
    ids = tokenize(base, filler)
    out = []
    while len(out) < n:
        out.extend(ids)
    return out[:n]


def run_completion(base, prompt_tokens, label):
    body = {
        "prompt": prompt_tokens,
        "n_predict": 16,
        "ignore_eos": True,
        "cache_prompt": True,
        "stream": False,
    }
    t0 = time.time()
    j = post(base, "/completion", body)
    wall = time.time() - t0
    tm = j.get("timings", {})
    prompt_n = tm.get("prompt_n")
    prompt_ms = tm.get("prompt_ms")
    print(f"\n[{label}]  prompt enviado = {len(prompt_tokens)} tokens")
    print(f"  timings.prompt_n  = {prompt_n}   (tokens REALMENTE prefilados)")
    print(f"  timings.prompt_ms = {prompt_ms:.0f} ms" if prompt_ms else "  timings.prompt_ms = ?")
    print(f"  wall              = {wall:.1f} s")
    print(f"  tokens_cached(usage) = {j.get('tokens_evaluated')} | truncated={j.get('truncated')}")
    return prompt_n, prompt_ms, len(prompt_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:1234")
    ap.add_argument("--n", type=int, default=50000, help="tamanho do prompt base")
    ap.add_argument("--append", type=int, default=64, help="tokens novos no 2º turno")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    props = get(base, "/props")
    n_ctx = props["default_generation_settings"]["n_ctx"]
    print(f"n_ctx={n_ctx}  build={props.get('build_info')}")
    print(f"Base de {args.n} tokens; 2º turno estende o MESMO prefixo + {args.append} tokens novos.")
    print("Se o reuso funciona, o 2º prefill processa ~poucos tokens, não o prompt inteiro.")

    base_tokens = build_tokens(base, args.n)

    # Turno 1: prefill frio do prefixo.
    p1_n, p1_ms, _ = run_completion(base, base_tokens, "TURNO 1 — frio")

    # Turno 2: MESMO prefixo + tokens novos no fim (simula anexar tool output).
    extra = build_tokens(base, args.n + args.append)[args.n:]
    turn2 = base_tokens + extra
    p2_n, p2_ms, total2 = run_completion(base, turn2, "TURNO 2 — estende prefixo")

    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)
    if p2_n is None:
        print("  Sem timings — não consigo concluir.")
        return
    reuse_ratio = 1 - (p2_n / total2)
    print(f"  Turno 2 enviou {total2} tokens, mas prefilou apenas {p2_n}.")
    print(f"  Reaproveitamento do KV: {reuse_ratio*100:.1f}% do prefixo.")
    if p2_n <= args.append * 4:
        print("  ✅ REUSO FUNCIONA — checkpoints+cache-ram evitam o re-prefill total.")
        print("     ctx 131072 fica viável: turnos seguintes são baratos.")
    elif p2_n < total2 * 0.5:
        print("  🟡 REUSO PARCIAL — economiza, mas ainda re-processa um bom pedaço.")
    else:
        print("  ❌ FULL REPROCESS — o KV NÃO foi reaproveitado (problema do SSM persiste).")
        print("     Cada turno volta a custar o prefill inteiro (~190s a 131k).")
        if p1_ms and p2_ms:
            print(f"     prefill turno1={p1_ms:.0f}ms  turno2={p2_ms:.0f}ms (deviam diferir muito se houvesse reuso)")


if __name__ == "__main__":
    main()
