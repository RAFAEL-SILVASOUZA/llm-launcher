"""Rodada com thinking DESLIGADO. Compara latência e qualidade."""
import json
import time
import urllib.request
import re
import textwrap

ENDPOINT = "http://192.168.18.223:1234/v1/chat/completions"
MODEL = "Darwin-36B-Opus.Q4_K_M"
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def call(messages, max_tokens=800, temperature=0.1, no_think=True):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if no_think:
        payload.update(NO_THINK)
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read().decode("utf-8")
    el = time.perf_counter() - t0
    j = json.loads(body)
    msg = j["choices"][0]["message"]
    timings = j.get("timings", {})
    usage = j.get("usage", {})
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "elapsed": el,
        "ct": usage.get("completion_tokens", 0),
        "pt": usage.get("prompt_tokens", 0),
        "decode_tps": timings.get("predicted_per_second", 0.0),
        "finish": j["choices"][0].get("finish_reason"),
    }


tests = [
    ("warmup",
        [{"role": "user", "content": "Responda apenas: ok"}], 100),
    ("math (no-think)",
        [{"role": "user", "content": textwrap.dedent("""\
            Uma caixa tem moedas de 1 e 5 centavos, total R$ 1,69 em 65 moedas.
            Quantas moedas de 5 centavos há? Termine com:
            RESPOSTA: <numero>
        """)}], 600),
    ("json (no-think)",
        [{"role": "system", "content": "Responda APENAS com JSON válido, sem markdown."},
         {"role": "user", "content":
            "JSON descrevendo Python com chaves: nome, linguagem, ano (int), "
            "criador, paradigmas (lista)."}], 400),
    ("lógica (no-think)",
        [{"role": "user", "content": textwrap.dedent("""\
            A diz: "B é o mentiroso." B diz: "A diz a verdade."
            C diz: "Eu não sou o mentiroso." Exatamente um mente.
            MENTIROSO: <A|B|C>
        """)}], 500),
    ("código (no-think)",
        [{"role": "user", "content":
            "Implemente em Python merge_intervals(intervals): lista de [start,end] "
            "inclusivos, devolve mesclada. Só o código em ```python."}], 800),
]


OK_MARK = "[OK]"
FAIL_MARK = "[X]"


def grade_math(c):
    m = re.search(r"RESPOSTA:\s*(\d+)", c, re.I)
    if m:
        return f"RESPOSTA={m.group(1)} {OK_MARK if int(m.group(1))==26 else FAIL_MARK}"
    if "26" in c:
        return f"contem 26 {OK_MARK}"
    return f"nao achou {FAIL_MARK}"


def grade_json(c):
    m = re.search(r"\{[\s\S]*\}", c)
    if not m:
        return f"sem JSON {FAIL_MARK}"
    try:
        o = json.loads(m.group(0))
    except Exception as e:
        return f"JSON invalido: {e} {FAIL_MARK}"
    req = ["nome", "linguagem", "ano", "criador", "paradigmas"]
    miss = [k for k in req if k not in o]
    if miss:
        return f"faltam {miss} {FAIL_MARK}"
    return f"OK (ano={o.get('ano')}, {len(o.get('paradigmas', []))} paradigmas) {OK_MARK}"


def grade_logic(c):
    m = re.search(r"MENTIROSO:\s*([ABC])", c, re.I)
    if m:
        return f"MENTIROSO={m.group(1)} {OK_MARK if m.group(1).upper()=='C' else FAIL_MARK}"
    return f"sem tag {FAIL_MARK}"


def grade_code(c):
    m = re.search(r"```(?:python)?\s*(.*?)```", c, re.DOTALL)
    if not m:
        return f"sem bloco {FAIL_MARK}"
    code = m.group(1)
    try:
        ns = {}
        exec(code, ns)
        fn = ns.get("merge_intervals")
        if not fn:
            return f"sem funcao {FAIL_MARK}"
        cases = [
            ([[1,3],[2,6],[8,10],[15,18]], [[1,6],[8,10],[15,18]]),
            ([[1,4],[4,5]], [[1,5]]),
            ([], []),
            ([[1,4],[0,4]], [[0,4]]),
        ]
        passed = sum(1 for inp, exp in cases if fn([list(x) for x in inp]) == exp)
        return f"{passed}/{len(cases)} casos {OK_MARK if passed==len(cases) else FAIL_MARK}"
    except Exception as e:
        return f"exec: {e} {FAIL_MARK}"


graders = {
    "math (no-think)": grade_math,
    "json (no-think)": grade_json,
    "lógica (no-think)": grade_logic,
    "código (no-think)": grade_code,
}

print(f"{'teste':<22}{'tempo':>7}{'ct':>6}{'reas':>6}{'cont':>6}{'dec t/s':>9}  grade")
print("-" * 85)
for name, msgs, mt in tests:
    r = call(msgs, max_tokens=mt)
    g = graders.get(name, lambda c: f"len={len(c)}")(r["content"]) if r["content"] else "content vazio"
    print(f"{name:<22}{r['elapsed']:>6.2f}s{r['ct']:>6}"
          f"{len(r['reasoning']):>6}{len(r['content']):>6}{r['decode_tps']:>9.1f}  {g}")
    if r["content"]:
        first = r["content"].strip().splitlines()[0][:90]
        print(f"  >> {first}")
