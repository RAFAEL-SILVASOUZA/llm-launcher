"""Testa se dá para desligar reasoning no Darwin-36B."""
import json
import urllib.request
import time

ENDPOINT = "http://192.168.18.223:1234/v1/chat/completions"
MODEL = "Darwin-36B-Opus.Q4_K_M"


def call(messages, extra=None, max_tokens=400):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read().decode("utf-8")
    elapsed = time.perf_counter() - t0
    j = json.loads(body)
    msg = j["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "elapsed": elapsed,
        "finish": j["choices"][0].get("finish_reason"),
    }


tries = [
    ("baseline",
        [{"role": "user", "content": "Diga 'oi' em português, em uma palavra."}],
        None),
    ("/no_think tag",
        [{"role": "user", "content": "Diga 'oi' em português, em uma palavra. /no_think"}],
        None),
    ("system /no_think",
        [{"role": "system", "content": "/no_think"},
         {"role": "user", "content": "Diga 'oi' em português, em uma palavra."}],
        None),
    ("enable_thinking=false",
        [{"role": "user", "content": "Diga 'oi' em português, em uma palavra."}],
        {"enable_thinking": False}),
    ("chat_template_kwargs",
        [{"role": "user", "content": "Diga 'oi' em português, em uma palavra."}],
        {"chat_template_kwargs": {"enable_thinking": False}}),
    ("reasoning_effort=none",
        [{"role": "user", "content": "Diga 'oi' em português, em uma palavra."}],
        {"reasoning_effort": "none"}),
]

for name, msgs, extra in tries:
    try:
        r = call(msgs, extra)
        print(f"\n[{name}] elapsed={r['elapsed']:.2f}s finish={r['finish']}")
        print(f"  reasoning_len={len(r['reasoning'])}  content_len={len(r['content'])}")
        if r["reasoning"]:
            print(f"  reasoning[:60]: {r['reasoning'][:60]!r}")
        if r["content"]:
            print(f"  content: {r['content'][:80]!r}")
    except Exception as e:
        print(f"\n[{name}] ERRO: {e}")
