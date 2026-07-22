"""Parser de metadados GGUF — extrai arch, layers, head dims, SSM/MoE/SWA, splits.

Portado de _gguf_meta em models.py. Lê só os campos necessários pro estimador
de memória (não decodifica vocabs gigantes). Cache em memória por path.
"""
import re
import struct
from pathlib import Path

_MIB = 1024 * 1024

# Bytes por elemento dos tensores GGUF, por ggml_type. Quants usam blocos com
# (block_size_bytes / n_elems_per_block).
GGML_BYTES_PER_ELEM = {
    0:  4.0,         # F32
    1:  2.0,         # F16
    2:  18/32,       # Q4_0
    3:  20/32,       # Q4_1
    6:  22/32,       # Q5_0
    7:  24/32,       # Q5_1
    8:  34/32,       # Q8_0
    9:  40/32,       # Q8_1
    10: 84/256,      # Q2_K
    11: 110/256,     # Q3_K
    12: 144/256,     # Q4_K
    13: 176/256,     # Q5_K
    14: 210/256,     # Q6_K
    15: 292/256,     # Q8_K
    16: 102/256,     # IQ2_XXS
    17: 110/256,     # IQ2_XS
    18: 174/256,     # IQ3_XXS
    19: 56/256,      # IQ1_S
    20: 144/256,     # IQ4_NL
    21: 132/256,     # IQ3_S
    22: 132/256,     # IQ2_S
    23: 144/256,     # IQ4_XS
    24: 1.0,         # I8
    25: 2.0,         # I16
    26: 4.0,         # I32
    27: 8.0,         # I64
    28: 8.0,         # F64
    29: 64/256,      # IQ1_M
    30: 2.0,         # BF16
    34: 54/256,      # TQ1_0
    35: 66/256,      # TQ2_0
    39: 17/32,       # MXFP4
    40: 18/32,       # NVFP4
    41: 12/256,      # Q1_0
}

_MOE_TENSOR_RE   = re.compile(r"blk\.(\d+)\.[^/]*?(?:_exps|\.exps)", re.IGNORECASE)
_ATTN_K_TENSOR_RE = re.compile(r"blk\.(\d+)\.attn_k", re.IGNORECASE)
_TOKEN_EMBD_RE   = re.compile(r"^token_embd\.", re.IGNORECASE)
_OUTPUT_RE       = re.compile(r"^output\.weight$", re.IGNORECASE)
_BLOCK_RE        = re.compile(r"^blk\.(\d+)\.", re.IGNORECASE)

_GGUF_META_CACHE: dict[str, dict] = {}


def read_meta(model_path: Path) -> dict | None:
    """Lê metadados do GGUF. Devolve None se não conseguiu parsear.

    Sempre percorre tensor_info pra coletar:
      - peso médio dos `ffn_*_exps` (alimenta sugestão de --n-cpu-moe)
      - quais camadas têm atenção plena (híbridos SSM)
      - tamanho exato de token_embd / output / blocks (alimenta split GPU/CPU)
    """
    key = str(model_path)
    if key in _GGUF_META_CACHE:
        return _GGUF_META_CACHE[key] or None

    raw: dict = {}
    moe_per_layer_bytes: dict[int, float] = {}
    attn_layer_set: set[int] = set()
    token_embd_bytes  = 0.0
    output_bytes      = 0.0
    blocks_bytes      = 0.0
    other_bytes       = 0.0

    try:
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                _GGUF_META_CACHE[key] = {}
                return None
            f.read(4)                                              # version
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            kv_count  = struct.unpack("<Q", f.read(8))[0]

            def read_value(vtype: int):
                if vtype == 0:  return f.read(1)[0]
                if vtype == 1:  return struct.unpack("<b", f.read(1))[0]
                if vtype == 2:  return struct.unpack("<H", f.read(2))[0]
                if vtype == 3:  return struct.unpack("<h", f.read(2))[0]
                if vtype == 4:  return struct.unpack("<I", f.read(4))[0]
                if vtype == 5:  return struct.unpack("<i", f.read(4))[0]
                if vtype == 6:  return struct.unpack("<f", f.read(4))[0]
                if vtype == 7:  return bool(f.read(1)[0])
                if vtype == 10: return struct.unpack("<Q", f.read(8))[0]
                if vtype == 11: return struct.unpack("<q", f.read(8))[0]
                if vtype == 12: return struct.unpack("<d", f.read(8))[0]
                if vtype == 8:
                    sl = struct.unpack("<Q", f.read(8))[0]
                    return f.read(sl).decode("utf-8", errors="replace")
                if vtype == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    n  = struct.unpack("<Q", f.read(8))[0]
                    if n <= 4096 and et in (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12):
                        return [read_value(et) for _ in range(n)]
                    for _ in range(n):
                        read_value(et)
                    return None
                raise ValueError(f"unknown GGUF type {vtype}")

            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                k_str = f.read(key_len).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]
                raw[k_str] = read_value(vtype)

            for _ in range(n_tensors):
                nl = struct.unpack("<Q", f.read(8))[0]
                tname = f.read(nl).decode("utf-8", errors="replace")
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ttype = struct.unpack("<I", f.read(4))[0]
                f.read(8)                                      # offset

                elems = 1
                for d in dims:
                    elems *= d
                bpe = GGML_BYTES_PER_ELEM.get(ttype, 1.0)
                t_bytes = elems * bpe

                am = _ATTN_K_TENSOR_RE.search(tname)
                if am:
                    attn_layer_set.add(int(am.group(1)))

                if _TOKEN_EMBD_RE.match(tname):
                    token_embd_bytes += t_bytes
                elif _OUTPUT_RE.match(tname):
                    output_bytes += t_bytes
                elif _BLOCK_RE.match(tname):
                    blocks_bytes += t_bytes
                    m = _MOE_TENSOR_RE.search(tname)
                    if m:
                        li = int(m.group(1))
                        moe_per_layer_bytes[li] = moe_per_layer_bytes.get(li, 0.0) + t_bytes
                else:
                    other_bytes += t_bytes
    except Exception:
        _GGUF_META_CACHE[key] = {}
        return None

    arch = raw.get("general.architecture")
    if not arch:
        _GGUF_META_CACHE[key] = {}
        return None

    def g(name, default=None):
        v = raw.get(f"{arch}.{name}")
        return default if v is None else v

    n_ctx_train = g("context_length")
    n_layer = g("block_count")
    n_embd  = g("embedding_length")
    n_head  = g("attention.head_count")
    n_kv    = g("attention.head_count_kv", n_head)
    key_len = g("attention.key_length")
    val_len = g("attention.value_length")

    swa_window     = g("attention.sliding_window")
    swa_pattern    = g("attention.sliding_window_pattern")
    key_len_swa    = g("attention.key_length_swa")
    val_len_swa    = g("attention.value_length_swa")

    ssm_inner = g("ssm.inner_size")
    ssm_state = g("ssm.state_size")
    ssm_conv  = g("ssm.conv_kernel")
    ssm_group = g("ssm.group_count")
    ssm_state_per_layer_mib = 0.0
    if ssm_inner and ssm_state and ssm_conv:
        conv_elems  = (int(ssm_inner) + 2 * int(ssm_group or 0) * int(ssm_state)) * (int(ssm_conv) - 1)
        state_elems = int(ssm_inner) * int(ssm_state)
        ssm_state_per_layer_mib = (conv_elems + state_elems) * 4 / _MIB

    if not (n_layer and n_embd and n_head):
        _GGUF_META_CACHE[key] = {}
        return None

    # Modelos de atenção híbrida (Laguna etc.) gravam attention.head_count como
    # array (um valor por camada). Downstream só quer um escalar representativo;
    # max() capta o pior caso (GQA e fallback de head_dim), igual ao que já se
    # faz com n_head_kv logo abaixo.
    n_head_typ = max(n_head) if isinstance(n_head, list) else int(n_head)

    head_dim_k = key_len or (n_embd // n_head_typ)
    head_dim_v = val_len or head_dim_k

    if isinstance(n_kv, list):
        n_kv_norm: int | list = [int(x) for x in n_kv]
    else:
        n_kv_norm = int(n_kv)

    expert_count        = int(g("expert_count", 0) or 0)
    expert_used_count   = int(g("expert_used_count", 0) or 0)
    moe_layers_count    = len(moe_per_layer_bytes)
    moe_per_layer_mib   = 0
    if moe_layers_count:
        avg_bytes = sum(moe_per_layer_bytes.values()) / moe_layers_count
        moe_per_layer_mib = int(avg_bytes / _MIB)

    chat_template = raw.get("tokenizer.chat_template") or ""
    tpl_lower     = chat_template.lower()
    # Fonte de verdade pra "o modelo pensa": o próprio template. Um modelo de
    # raciocínio precisa abrir a resposta com <think> (ou expor enable_thinking /
    # reasoning_content), então isso está sempre no template — enquanto o nome do
    # arquivo não diz nada (ornith-1.0-35b não tem nenhuma palavra reveladora).
    has_thinking = any(
        kw in tpl_lower for kw in ("<think>", "enable_thinking", "reasoning_content")
    )

    token_embd_mib  = int(token_embd_bytes  / _MIB)
    output_mib      = int(output_bytes      / _MIB)
    blocks_mib      = int(blocks_bytes      / _MIB)
    tensor_total_mib = token_embd_mib + output_mib + blocks_mib + int(other_bytes / _MIB)

    meta = {
        "arch":              arch,
        # Contexto de treino: teto acima do qual sugerir ctx é pedir degradação.
        "n_ctx_train":       int(n_ctx_train) if n_ctx_train else None,
        "n_layer":           int(n_layer),
        "n_embd":            int(n_embd),
        "n_head":            n_head_typ,
        "n_head_kv":         n_kv_norm,
        "head_dim_k":        int(head_dim_k),
        "head_dim_v":        int(head_dim_v),
        "swa_window":        int(swa_window) if swa_window else None,
        "swa_pattern":       [bool(x) for x in swa_pattern] if isinstance(swa_pattern, list) else None,
        "head_dim_k_swa":    int(key_len_swa) if key_len_swa else None,
        "head_dim_v_swa":    int(val_len_swa) if val_len_swa else None,
        "is_moe":            expert_count > 0,
        "expert_count":      expert_count,
        "expert_used_count": expert_used_count,
        "moe_layers_count":  moe_layers_count,
        "moe_per_layer_mib": moe_per_layer_mib,
        "token_embd_mib":    token_embd_mib,
        "output_mib":        output_mib,
        "blocks_mib":        blocks_mib,
        "tensor_total_mib":  tensor_total_mib,
        "attn_layers":       sorted(attn_layer_set),
        "ssm_state_per_layer_mib": ssm_state_per_layer_mib,
        "supports_preserve_thinking": "preserve_thinking" in chat_template,
        "has_thinking_template":      has_thinking,
    }
    _GGUF_META_CACHE[key] = meta
    return meta
