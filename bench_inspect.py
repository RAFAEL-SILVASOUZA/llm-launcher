import json
import urllib.request

ENDPOINT = "http://192.168.18.223:1234/v1/chat/completions"
payload = {
    "model": "Darwin-36B-Opus.Q4_K_M",
    "messages": [{"role": "user", "content": "Diga em uma frase: o que é Python?"}],
    "max_tokens": 80,
    "temperature": 0.0,
    "stream": False,
}
req = urllib.request.Request(
    ENDPOINT,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=60) as r:
    body = r.read().decode("utf-8")
print(body)
