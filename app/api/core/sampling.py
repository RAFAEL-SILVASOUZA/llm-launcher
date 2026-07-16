"""Samplers por modelo — resolve temp/top-p/top-k/min-p/repeat-penalty e a PROCEDÊNCIA.

O problema que isto resolve: um preset único hardcoded pra todo modelo. O preset
antigo (0.3 / 0.8 / repeat 1.05) foi calibrado em A/B pra agente de código com
modelo NÃO-thinking, e era servido também pra modelo de raciocínio — onde
temperatura baixa + top-p apertado estreita a cadeia de pensamento (mais loop
dentro do <think>, não menos) e repeat-penalty pune a repetição legítima de
fórmula/identificador. Card do Qwen3/QwQ/Ornith pede 0.6 / 0.95 sem penalidade.

Fontes, da mais confiável pra menos:

  manual             o usuário editou na UI. Nunca é sobrescrito por nada.
  generation_config  os valores que o AUTOR publicou (generation_config.json do
                     repo HF, gravado como sidecar sampling.json no download).
  template           derivado do chat template do GGUF: pensa → preset reasoning,
                     não pensa → preset código. Certeza sobre a CLASSE, não sobre
                     os números.
  default            nenhum sinal (GGUF ilegível) — preset código, conservador.

A UI mostra a procedência justamente pra "default" e "template" não passarem por
verdade publicada pelo autor.
"""
import json
from pathlib import Path

from . import gguf

SIDECAR_NAME = "sampling.json"

SAMPLER_KEYS = ("temp", "top_p", "top_k", "min_p", "repeat_penalty")

# Não-thinking / agente de código: preset validado em A/B (2026-06-11, 2026-06-17).
# min_p 0.05 é o default do próprio llama.cpp — explicitado aqui só pra não mudar
# em silêncio o comportamento que foi validado.
PRESET_CODE = {
    "temp": 0.3, "top_p": 0.8, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.05,
}
# Reasoning: o que os cards pedem. repeat_penalty 1.0 = desligado; min_p 0 porque o
# default 0.05 do llama.cpp corta a cauda que o top-p 0.95 justamente quer manter.
PRESET_REASONING = {
    "temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
}

PRESETS = {"code": PRESET_CODE, "reasoning": PRESET_REASONING}


def preset_for(thinking: bool) -> dict:
    return dict(PRESET_REASONING if thinking else PRESET_CODE)


def from_generation_config(gc: dict) -> dict | None:
    """Normaliza um generation_config.json do HF pro nosso dict. None se não houver
    nada de sampling ali (muitos só têm eos_token_id/bos_token_id — isso não conta).

    `do_sample: false` significa greedy: o autor não publicou preferência de
    temperatura, então não temos o que extrair.
    """
    if not isinstance(gc, dict) or gc.get("do_sample") is False:
        return None

    mapping = {
        "temp":           ("temperature",),
        "top_p":          ("top_p",),
        "top_k":          ("top_k",),
        "min_p":          ("min_p",),
        "repeat_penalty": ("repetition_penalty", "repeat_penalty"),
    }
    out: dict = {}
    for key, names in mapping.items():
        for name in names:
            val = gc.get(name)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out[key] = int(val) if key == "top_k" else float(val)
                break

    # Sem temperatura não é config de sampling — é só metadado de tokens.
    if "temp" not in out:
        return None
    return out


def sidecar_path(model_path: Path) -> Path:
    return model_path.parent / SIDECAR_NAME


def read_sidecar(model_path: Path) -> dict | None:
    """Lê o sampling.json gravado no download. Procura na pasta do .gguf e na
    pasta-pai — repos de quant guardam cada quant num subdiretório."""
    for folder in (model_path.parent, model_path.parent.parent):
        f = folder / SIDECAR_NAME
        try:
            if f.is_file():
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict) and any(k in data for k in SAMPLER_KEYS):
                    return data
        except Exception:
            continue
    return None


def write_sidecar(folder: Path, payload: dict) -> Path | None:
    """Grava o sampling.json ao lado do modelo. Best-effort — falhar aqui nunca
    pode derrubar um download que já terminou."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / SIDECAR_NAME
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return dest
    except Exception:
        return None


def is_thinking(model_path: Path) -> bool:
    """Wrapper pra manter a detecção num lugar só (o chat template do GGUF)."""
    meta = gguf.read_meta(model_path)
    if meta is None:
        return False
    return bool(meta.get("has_thinking_template"))


def resolve(model_path: Path, cfg: dict | None = None) -> dict:
    """Devolve {temp, top_p, top_k, min_p, repeat_penalty, source, thinking}.

    `cfg` é a config salva: se tiver sampler_source == "manual", os valores dela
    ganham de tudo (é o usuário mandando). Qualquer outro sampler_source é tratado
    como cache do que já resolvemos antes e re-derivado do zero — assim um modelo
    salvo antes deste código não fica preso ao preset errado.
    """
    thinking = is_thinking(model_path)

    if cfg and cfg.get("sampler_source") == "manual":
        out = preset_for(thinking)
        for k in SAMPLER_KEYS:
            val = cfg.get(k)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out[k] = val
        out["source"] = "manual"
        out["thinking"] = thinking
        return out

    side = read_sidecar(model_path)
    if side:
        out = preset_for(thinking)
        got = False
        for k in SAMPLER_KEYS:
            val = side.get(k)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out[k] = val
                got = True
        if got:
            out["source"] = side.get("source") or "generation_config"
            out["from_repo"] = side.get("from_repo")
            out["thinking"] = thinking
            return out

    meta_ok = gguf.read_meta(model_path) is not None
    out = preset_for(thinking)
    out["source"] = "template" if meta_ok else "default"
    out["thinking"] = thinking
    return out


def _fmt(v) -> str:
    if isinstance(v, int):
        return str(v)
    # 0.3 -> "0.3", não "0.30000000000000004"
    return f"{float(v):g}"


def cli_flags(s: dict) -> str:
    """Trecho de linha de comando. Emite as cinco flags SEMPRE, mesmo as que
    coincidem com o default do llama.cpp — o preview do comando é onde o usuário
    confere o que o modelo vai receber, e valor implícito não se confere."""
    return (
        f" --temp {_fmt(s['temp'])}"
        f" --top-p {_fmt(s['top_p'])}"
        f" --top-k {_fmt(s['top_k'])}"
        f" --min-p {_fmt(s['min_p'])}"
        f" --repeat-penalty {_fmt(s['repeat_penalty'])}"
    )


def ini_pairs(s: dict) -> list[tuple[str, str]]:
    """Mesmas flags em chave/valor pro INI do modo router."""
    return [
        ("temp",           _fmt(s["temp"])),
        ("top-p",          _fmt(s["top_p"])),
        ("top-k",          _fmt(s["top_k"])),
        ("min-p",          _fmt(s["min_p"])),
        ("repeat-penalty", _fmt(s["repeat_penalty"])),
    ]
