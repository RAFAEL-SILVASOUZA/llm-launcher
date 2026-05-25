"""Benchmark Darwin-36B-Opus.Q4_K_M (reasoning model).

Captura reasoning_content + content. max_tokens generoso para o modelo
chegar a emitir resposta final depois do raciocínio.
"""
import json
import time
import urllib.request
import re
import textwrap
from dataclasses import dataclass, field

ENDPOINT = "http://192.168.18.223:1234/v1/chat/completions"
MODEL = "Darwin-36B-Opus.Q4_K_M"
TIMEOUT = 600


@dataclass
class Result:
    name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_chars: int = 0
    content_chars: int = 0
    elapsed_s: float = 0.0
    decode_tps: float = 0.0
    prefill_tps: float = 0.0
    grade: str = ""
    notes: list = field(default_factory=list)
    reasoning: str = ""
    content: str = ""
    finish: str = ""


def chat(messages, *, max_tokens=2048, temperature=0.2, top_p=0.9, stop=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if stop:
        payload["stop"] = stop
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        body = r.read().decode("utf-8")
    elapsed = time.perf_counter() - t0
    resp = json.loads(body)
    msg = resp["choices"][0]["message"]
    finish = resp["choices"][0].get("finish_reason", "")
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = resp.get("usage", {})
    timings = resp.get("timings", {})
    return content, reasoning, elapsed, usage, timings, finish


def run(name, messages, *, max_tokens=2048, temperature=0.2, grader=None):
    print(f"\n=== {name} ===")
    try:
        content, reasoning, elapsed, usage, timings, finish = chat(
            messages, max_tokens=max_tokens, temperature=temperature
        )
    except Exception as e:
        print(f"  ERRO: {e}")
        r = Result(name=name)
        r.notes.append(f"ERRO: {e}")
        return r

    r = Result(
        name=name,
        prompt_tokens=usage.get("prompt_tokens", 0) or 0,
        completion_tokens=usage.get("completion_tokens", 0) or 0,
        reasoning_chars=len(reasoning),
        content_chars=len(content),
        elapsed_s=elapsed,
        decode_tps=timings.get("predicted_per_second", 0.0) or 0.0,
        prefill_tps=timings.get("prompt_per_second", 0.0) or 0.0,
        reasoning=reasoning,
        content=content,
        finish=finish,
    )
    if grader:
        r.grade, r.notes = grader(content, reasoning)
    print(
        f"  tempo={elapsed:.2f}s  pt={r.prompt_tokens} ct={r.completion_tokens}  "
        f"prefill={r.prefill_tps:.0f}t/s decode={r.decode_tps:.1f}t/s  finish={finish}"
    )
    print(f"  reasoning={r.reasoning_chars} chars  content={r.content_chars} chars")
    if r.grade:
        print(f"  nota: {r.grade}")
    for n in r.notes:
        print(f"   - {n}")
    if content:
        first = content.strip().splitlines()[0][:140]
        print(f"  >> {first}")
    elif reasoning:
        print("  (sem content; finish=length — reasoning truncado)")
    return r


# ---------------- Graders ----------------

def g_math(content, reasoning):
    notes = []
    text = (content or "") + "\n" + (reasoning or "")
    # Resposta correta: 26 moedas de 5 centavos.
    m = re.search(r"RESPOSTA:\s*(\d+)", content, re.I)
    if m:
        ans = int(m.group(1))
        notes.append(f"RESPOSTA capturada: {ans}")
        return ("OK" if ans == 26 else "FAIL"), notes
    if re.search(r"\b26\b", content):
        notes.append("contém 26 em content")
        return "OK", notes
    if re.search(r"\b26\b", reasoning):
        notes.append("acertou em reasoning mas content vazio")
        return "PARTIAL", notes
    return "FAIL", notes


def g_code(content, reasoning):
    notes = []
    text = content or ""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if not m:
        # try reasoning as fallback (algumas vezes o código sai no thinking)
        m = re.search(r"```(?:python)?\s*(.*?)```", reasoning or "", re.DOTALL)
        if m:
            notes.append("código veio em reasoning")
        else:
            return "FAIL", ["sem bloco python"]
    code = m.group(1)
    notes.append(f"linhas: {len(code.splitlines())}")
    try:
        ns = {}
        exec(code, ns)
        fn = ns.get("merge_intervals")
        if not callable(fn):
            return "FAIL", notes + ["função merge_intervals ausente"]
        cases = [
            ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
            ([[1, 4], [4, 5]], [[1, 5]]),
            ([], []),
            ([[1, 4], [0, 4]], [[0, 4]]),
            ([[1, 4], [2, 3]], [[1, 4]]),
            ([[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]),
        ]
        passed = 0
        for inp, exp in cases:
            try:
                got = fn([list(x) for x in inp])
                if got == exp:
                    passed += 1
                else:
                    notes.append(f"falha {inp} -> {got} esperado {exp}")
            except Exception as e:
                notes.append(f"exc {inp}: {e}")
        notes.append(f"casos: {passed}/{len(cases)}")
        return ("OK" if passed == len(cases) else f"{passed}/{len(cases)}"), notes
    except Exception as e:
        return "FAIL", notes + [f"exec: {e}"]


def g_ptbr(content, reasoning):
    notes = []
    if not content:
        return "FAIL", ["content vazio (reasoning truncou orçamento)"]
    pt_markers = ["é", "ção", "não", "para", "uma", "que", "porque", "do", "da"]
    score = sum(1 for w in pt_markers if w.lower() in content.lower())
    en_leak = sum(1 for w in [" the ", " is ", " of ", " with "] if w in content.lower())
    notes.append(f"marcadores PT: {score}/{len(pt_markers)}  vazamento EN: {en_leak}")
    return ("OK" if score >= 5 and en_leak == 0 else "WARN"), notes


def g_json(content, reasoning):
    notes = []
    if not content:
        return "FAIL", ["content vazio"]
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return "FAIL", ["sem objeto JSON em content"]
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        return "FAIL", [f"JSON inválido: {e}"]
    req = ["nome", "linguagem", "ano", "criador", "paradigmas"]
    miss = [k for k in req if k not in obj]
    if miss:
        notes.append(f"chaves faltantes: {miss}")
    if not isinstance(obj.get("paradigmas"), list):
        notes.append("paradigmas não é lista")
    if obj.get("ano") != 1991:
        notes.append(f"ano={obj.get('ano')} (esperado 1991)")
    return ("OK" if not miss and isinstance(obj.get("paradigmas"), list) else "PARTIAL"), notes


def g_logic(content, reasoning):
    notes = []
    text = content or ""
    m = re.search(r"MENTIROSO:\s*([ABC])", text, re.I)
    if m:
        ans = m.group(1).upper()
        notes.append(f"MENTIROSO: {ans}")
        return ("OK" if ans == "C" else "FAIL"), notes
    # Resposta correta: C é o mentiroso.
    if re.search(r"\bC\b.{0,30}(mente|mentiroso)", text, re.I):
        return "OK", ["identifica C (sem tag formal)"]
    if reasoning and re.search(r"\bC\b.{0,30}(mente|mentiroso)", reasoning, re.I):
        return "PARTIAL", ["acertou em reasoning"]
    return "FAIL", notes + ["não identifica C"]


def g_needle(content, reasoning):
    notes = []
    target = "PURPLE-MONKEY-DISHWASHER-7331"
    if target in (content or ""):
        return "OK", ["recuperou em content"]
    if target in (reasoning or ""):
        return "PARTIAL", ["recuperou em reasoning"]
    return "FAIL", ["não recuperou"]


def g_simple(content, reasoning):
    # apenas mede se houve content
    if content:
        return "OK", [f"len={len(content)}"]
    return "FAIL", ["content vazio"]


# ---------------- Tests ----------------

def t_warmup():
    return run(
        "warmup",
        [{"role": "user", "content": "Responda apenas: ok"}],
        max_tokens=200, temperature=0.0,
    )


def t_throughput():
    msgs = [{"role": "user", "content":
        "Escreva 6 parágrafos explicando como funciona um Transformer (sem listas)."
    }]
    return run("throughput-6par", msgs, max_tokens=2000, temperature=0.5,
               grader=g_simple)


def t_math():
    msgs = [{"role": "user", "content": textwrap.dedent("""\
        Uma caixa contém moedas de 1 e 5 centavos, totalizando R$ 1,69 em
        65 moedas. Quantas moedas de 5 centavos há? Resolva e termine com
        a linha exata:
        RESPOSTA: <numero>
    """)}]
    return run("matemática", msgs, max_tokens=2500, temperature=0.1, grader=g_math)


def t_code():
    msgs = [{"role": "user", "content": textwrap.dedent("""\
        Implemente em Python `merge_intervals(intervals)`: recebe lista de
        [start, end] inclusivos, devolve lista mesclada e ordenada. Sem
        dependências externas. Devolva o código dentro de ```python ... ```.
    """)}]
    return run("código", msgs, max_tokens=3000, temperature=0.1, grader=g_code)


def t_ptbr():
    msgs = [{"role": "user", "content":
        "Em português brasileiro, explique em 4 frases por que Dijkstra "
        "não funciona com pesos negativos."
    }]
    return run("ptbr", msgs, max_tokens=2000, temperature=0.3, grader=g_ptbr)


def t_json():
    msgs = [
        {"role": "system", "content": "Responda APENAS com JSON válido, sem markdown e sem comentários."},
        {"role": "user", "content":
            "Gere um JSON descrevendo a linguagem Python com as chaves: "
            "nome, linguagem, ano (int), criador, paradigmas (lista)."
        },
    ]
    return run("json", msgs, max_tokens=2000, temperature=0.0, grader=g_json)


def t_logic():
    msgs = [{"role": "user", "content": textwrap.dedent("""\
        Três pessoas A, B e C. Exatamente uma sempre mente, as outras duas
        falam a verdade.
        A diz: "B é o mentiroso."
        B diz: "A está dizendo a verdade."
        C diz: "Eu não sou o mentiroso."
        Quem é o mentiroso? Termine com a linha exata:
        MENTIROSO: <A|B|C>
    """)}]
    return run("lógica", msgs, max_tokens=2500, temperature=0.1, grader=g_logic)


def t_needle():
    filler = "A história da computação tem muitas curiosidades. "
    needle = ("\n[NOTA CRUCIAL] A chave secreta é "
              "PURPLE-MONKEY-DISHWASHER-7331. Memorize.\n")
    pre = filler * 400
    post = filler * 400
    body = (
        "Leia o texto e devolva APENAS a chave secreta que aparece "
        "embutida nele.\n\n" + pre + needle + post +
        "\n\nQual é exatamente a chave secreta?"
    )
    return run("needle (~6k ctx)",
               [{"role": "user", "content": body}],
               max_tokens=1500, temperature=0.0, grader=g_needle)


def t_multiturn():
    msgs = [{"role": "user", "content": "Liste 5 algoritmos de ordenação."}]
    r1 = run("multi-1", msgs, max_tokens=1500, temperature=0.2, grader=g_simple)
    msgs.append({"role": "assistant", "content": r1.content})
    msgs.append({"role": "user", "content":
        "Pegue o terceiro da sua lista e diga a complexidade de pior caso."})
    r2 = run("multi-2", msgs, max_tokens=1500, temperature=0.2, grader=g_simple)
    return [r1, r2]


def main():
    print(f"Endpoint: {ENDPOINT}")
    print(f"Modelo:   {MODEL}")
    results = []
    results.append(t_warmup())
    results.append(t_throughput())
    results.append(t_math())
    results.append(t_code())
    results.append(t_ptbr())
    results.append(t_json())
    results.append(t_logic())
    results.append(t_needle())
    results.extend(t_multiturn())

    print("\n=========== RESUMO ===========")
    print(f"{'teste':<22}{'tempo':>8}{'pt':>7}{'ct':>7}{'reas':>6}{'cont':>6}"
          f"{'dec t/s':>9}{'nota':>11}")
    print("-" * 80)
    tot_ct = 0
    tot_t = 0.0
    for r in results:
        print(f"{r.name:<22}{r.elapsed_s:>7.2f}s{r.prompt_tokens:>7}"
              f"{r.completion_tokens:>7}{r.reasoning_chars:>6}{r.content_chars:>6}"
              f"{r.decode_tps:>9.1f}{(r.grade or '-'):>11}")
        tot_ct += r.completion_tokens
        tot_t += r.elapsed_s
    print("-" * 80)
    if tot_t > 0:
        print(f"TOTAL: {tot_ct} tok em {tot_t:.1f}s ({tot_ct/tot_t:.1f} t/s médio)")


if __name__ == "__main__":
    main()
