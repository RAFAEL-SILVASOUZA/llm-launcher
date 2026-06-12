#!/usr/bin/env python3
"""
test_ctx_exhaustion.py — Diagnostica a "interrupção silenciosa" do llama-server
quando a janela de contexto enche.

Alvo: o server MTP/Qwen3.6-27B levantado pelo models.py com --no-context-shift.
O design do launcher ASSUME que --no-context-shift devolve um "erro limpo" para o
agente compactar e re-tentar. Este script testa se isso é verdade ou se o server
apenas para de gerar em silêncio (finish_reason=stop/length sem aviso).

Só usa a stdlib (urllib + threading). Não escreve nada no servidor além de
requisições de inferência.

Uso:
    python test_ctx_exhaustion.py                 # roda todos os cenários
    python test_ctx_exhaustion.py --url http://localhost:1234
    python test_ctx_exhaustion.py --only overflow # só um cenário
"""

import argparse
import json
import sys
import threading
import time
import urllib.request
import urllib.error

# Console do Windows costuma ser cp1252; força UTF-8 pra não quebrar em "→"/acentos.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers (stdlib only)
# ──────────────────────────────────────────────────────────────────────────────

def _req(url, method="GET", body=None, timeout=600):
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def get_json(base, path):
    status, txt = _req(base + path)
    return json.loads(txt)


def tokenize(base, text):
    status, txt = _req(base + "/tokenize", "POST", {"content": text})
    return json.loads(txt)["tokens"]


def build_token_prompt(base, n_tokens):
    """Retorna uma lista de exatamente n_tokens ids, repetindo um filler neutro."""
    filler = ("The quick brown fox jumps over the lazy dog near the riverbank "
              "while the sun slowly sets behind the distant mountains. ")
    ids = tokenize(base, filler)
    if not ids:
        raise RuntimeError("tokenize devolveu vazio")
    out = []
    while len(out) < n_tokens:
        out.extend(ids)
    return out[:n_tokens]


# ──────────────────────────────────────────────────────────────────────────────
# Slot poller — captura n_past / estado do KV no momento da parada
# ──────────────────────────────────────────────────────────────────────────────

class SlotPoller(threading.Thread):
    def __init__(self, base, interval=0.25):
        super().__init__(daemon=True)
        self.base = base
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                slots = get_json(self.base, "/slots")
                s = slots[0] if isinstance(slots, list) and slots else {}
                nxt = s.get("next_token", [{}])
                nxt = nxt[0] if isinstance(nxt, list) and nxt else {}
                self.samples.append({
                    "t": round(time.time(), 3),
                    "is_processing": s.get("is_processing"),
                    "n_decoded": nxt.get("n_decoded"),
                    "n_remain": nxt.get("n_remain"),
                    "has_next_token": nxt.get("has_next_token"),
                })
            except Exception as e:
                self.samples.append({"t": round(time.time(), 3), "err": str(e)})
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def peak_decoded(self):
        vals = [s.get("n_decoded") for s in self.samples if isinstance(s.get("n_decoded"), int)]
        return max(vals) if vals else None


# ──────────────────────────────────────────────────────────────────────────────
# Cenários
# ──────────────────────────────────────────────────────────────────────────────

def scenario_prompt_overflow(base, n_ctx, model):
    """Prompt MAIOR que n_ctx. Esperado: erro HTTP limpo ANTES de gerar."""
    print("\n" + "=" * 78)
    print("CENÁRIO 1 — PROMPT OVERFLOW (prompt > n_ctx)")
    print("=" * 78)
    target = n_ctx + 2000
    print(f"Enviando prompt de {target} tokens (n_ctx={n_ctx}); n_predict=16")
    prompt = build_token_prompt(base, target)
    body = {"prompt": prompt, "n_predict": 16, "stream": False, "cache_prompt": False}
    t0 = time.time()
    try:
        status, txt = _req(base + "/completion", "POST", body)
        dt = time.time() - t0
        j = json.loads(txt)
        print(f"  HTTP {status} em {dt:.1f}s")
        print(f"  content='{j.get('content','')[:80]}'")
        print(f"  stop_type={j.get('stop_type')} stopped_limit={j.get('stopped_limit')}")
        print(f"  truncated={j.get('truncated')}")
        print(f"  tokens_evaluated={j.get('tokens_evaluated')} tokens_predicted={j.get('tokens_predicted')}")
        print("  >> VEREDICTO: server NÃO rejeitou — provável truncamento/shift silencioso")
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        err = e.read().decode("utf-8", "replace")
        print(f"  HTTP {e.code} em {dt:.1f}s (ERRO LIMPO)")
        print(f"  body: {err[:300]}")
        print("  >> VEREDICTO: erro limpo — agente CONSEGUE detectar e compactar")
    except Exception as e:
        print(f"  EXCEÇÃO: {type(e).__name__}: {e}")
        print("  >> VEREDICTO: falha de transporte (não é um erro de aplicação limpo)")


def scenario_generation_overflow(base, n_ctx, model, headroom=300):
    """Prompt logo abaixo de n_ctx + geração infinita (ignore_eos).
    Esperado: a geração bate no teto do contexto. Capturamos COMO ela para."""
    print("\n" + "=" * 78)
    print("CENÁRIO 2 — GENERATION OVERFLOW (prompt enche, geração estoura)")
    print("=" * 78)
    p_tokens = n_ctx - headroom
    print(f"Prompt de {p_tokens} tokens (n_ctx={n_ctx}, headroom={headroom})")
    print(f"n_predict=4096, ignore_eos=true → deve parar perto de {headroom} tokens gerados")
    prompt = build_token_prompt(base, p_tokens)
    body = {
        "prompt": prompt,
        "n_predict": 4096,
        "ignore_eos": True,
        "stream": False,
        "cache_prompt": False,
    }
    poller = SlotPoller(base)
    poller.start()
    t0 = time.time()
    try:
        status, txt = _req(base + "/completion", "POST", body)
        dt = time.time() - t0
        poller.stop(); poller.join(timeout=2)
        j = json.loads(txt)
        print(f"  HTTP {status} em {dt:.1f}s")
        print(f"  tokens_evaluated={j.get('tokens_evaluated')} tokens_predicted={j.get('tokens_predicted')}")
        print(f"  stop_type={j.get('stop_type')} stopped_eos={j.get('stopped_eos')} "
              f"stopped_word={j.get('stopped_word')} stopped_limit={j.get('stopped_limit')}")
        print(f"  truncated={j.get('truncated')}")
        te = j.get("tokens_evaluated") or 0
        tp = j.get("tokens_predicted") or 0
        print(f"  n_past final estimado = {te + tp} (teto = {n_ctx})")
        print(f"  pico n_decoded observado nos slots = {poller.peak_decoded()}")
        _verdict_generation(j, n_ctx, te, tp)
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        poller.stop(); poller.join(timeout=2)
        err = e.read().decode("utf-8", "replace")
        print(f"  HTTP {e.code} em {dt:.1f}s (ERRO durante geração)")
        print(f"  body: {err[:300]}")
        print(f"  pico n_decoded observado = {poller.peak_decoded()}")
        print("  >> VEREDICTO: server aborta com erro ao encher — visível ao cliente")
    except Exception as e:
        poller.stop(); poller.join(timeout=2)
        print(f"  EXCEÇÃO: {type(e).__name__}: {e}")


def _verdict_generation(j, n_ctx, te, tp):
    sl = j.get("stopped_limit")
    seos = j.get("stopped_eos")
    if sl:
        print("  >> VEREDICTO: parou por stopped_limit (bateu no contexto/n_predict).")
        print("     finish_reason='length' para o cliente OpenAI — detectável SE o")
        print("     cliente checar finish_reason. Muitos harnesses ignoram → parece silencioso.")
    elif seos:
        print("  >> VEREDICTO: parou por EOS apesar de ignore_eos — suspeito. "
              "Pode ser o template injetando <|im_end|> (vide stall SSM).")
    else:
        print("  >> VEREDICTO: parou sem stopped_limit/eos claro — ESTE é o modo silencioso.")


def scenario_streaming(base, n_ctx, model, headroom=300):
    """Mesma sobrecarga de geração, mas em streaming SSE — observa EXATAMENTE
    onde o stream termina e se há evento de erro ou só um fim abrupto."""
    print("\n" + "=" * 78)
    print("CENÁRIO 3 — STREAMING até o teto (observa o fim do SSE)")
    print("=" * 78)
    p_tokens = n_ctx - headroom
    prompt = build_token_prompt(base, p_tokens)
    body = {
        "prompt": prompt,
        "n_predict": 4096,
        "ignore_eos": True,
        "stream": True,
        "cache_prompt": False,
    }
    data = json.dumps(body).encode("utf-8")
    r = urllib.request.Request(base + "/completion", data=data, method="POST",
                               headers={"Content-Type": "application/json"})
    n_chunks = 0
    last = None
    first_t = None
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            print(f"  HTTP {resp.status}; lendo stream…")
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    print("  recebeu [DONE]")
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if first_t is None:
                    first_t = time.time() - t0
                n_chunks += 1
                last = ev
        dt = time.time() - t0
        print(f"  stream terminou: {n_chunks} chunks, 1º token em {first_t:.1f}s, total {dt:.1f}s")
        if last:
            print(f"  último evento: stop={last.get('stop')} "
                  f"stop_type={last.get('stop_type')} stopped_limit={last.get('stopped_limit')} "
                  f"stopped_eos={last.get('stopped_eos')}")
            print(f"  tokens_predicted={last.get('tokens_predicted')} "
                  f"tokens_evaluated={last.get('tokens_evaluated')}")
            if last.get("stopped_limit"):
                print("  >> VEREDICTO: stream fecha por LIMITE de contexto. O último chunk")
                print("     carrega stopped_limit=true, mas NÃO há mensagem de erro — um")
                print("     harness que só concatena 'content' vê a resposta parar do nada.")
            else:
                print("  >> VEREDICTO: ver flags acima; se nenhuma true, é parada silenciosa real.")
        else:
            print("  >> VEREDICTO: stream fechou SEM nenhum evento útil — parada totalmente silenciosa.")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode('utf-8','replace')[:300]}")
    except Exception as e:
        dt = time.time() - t0
        print(f"  EXCEÇÃO após {dt:.1f}s e {n_chunks} chunks: {type(e).__name__}: {e}")
        print("  >> VEREDICTO: conexão caiu no meio — o cliente vê EOF, não um erro de app.")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:1234")
    ap.add_argument("--only", choices=["overflow", "generation", "streaming"])
    ap.add_argument("--headroom", type=int, default=300)
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print("Sondando o servidor…")
    try:
        props = get_json(base, "/props")
        models = get_json(base, "/v1/models")
    except Exception as e:
        print(f"❌ Não consegui falar com {base}: {e}")
        sys.exit(1)

    n_ctx = props.get("default_generation_settings", {}).get("n_ctx") \
        or props.get("n_ctx")
    model = props.get("model_alias", "?")
    meta = models["data"][0].get("meta", {}) if models.get("data") else {}
    print(f"  modelo      : {model}")
    print(f"  build_info  : {props.get('build_info')}")
    print(f"  n_ctx       : {n_ctx}  (n_ctx_train={meta.get('n_ctx')})")
    print(f"  vision      : {props.get('modalities',{}).get('vision')}")
    print(f"  total_slots : {props.get('total_slots')}")
    print("\nLembrete: este server roda com --no-context-shift (vide models.py).")
    print("A pergunta do teste: encher o contexto vira erro LIMPO ou parada SILENCIOSA?")

    if args.only in (None, "overflow"):
        scenario_prompt_overflow(base, n_ctx, model)
    if args.only in (None, "generation"):
        scenario_generation_overflow(base, n_ctx, model, args.headroom)
    if args.only in (None, "streaming"):
        scenario_streaming(base, n_ctx, model, args.headroom)

    print("\n" + "=" * 78)
    print("FIM. Leia os VEREDICTOS acima. Cruzar com o log do llama-server (stderr):")
    print("  procure por 'context shift is disabled', 'n_past', 'context full',")
    print("  'common_context_can_seq_rm', 'slot released'.")
    print("=" * 78)


if __name__ == "__main__":
    main()
