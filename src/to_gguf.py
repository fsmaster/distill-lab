"""Minimal HF -> GGUF converter for our three students (Llama-arch SmolLM2 / tiny30, and Qwen3),
then quantize with llama.cpp's own quantizer via llama-cpp-python.

  python to_gguf.py work/models/smol135 site/models/smol135.gguf q8_0
"""
import sys, json, ctypes, pathlib, os
import numpy as np, torch, gguf
from safetensors.torch import load_file
import llama_cpp

src, out, qtype = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else "q8_0")
cfg = json.loads((src / "config.json").read_text())
arch = {"LlamaForCausalLM": "llama", "Qwen3ForCausalLM": "qwen3"}[cfg["architectures"][0]]
tj = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
tc = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
n_head, n_kv = cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_head

tmp = out.with_suffix(".f16.gguf"); out.parent.mkdir(parents=True, exist_ok=True)
w = gguf.GGUFWriter(str(tmp), arch)
w.add_name(src.name)
w.add_context_length(min(cfg.get("max_position_embeddings", 2048), 8192))
w.add_embedding_length(cfg["hidden_size"]); w.add_block_count(cfg["num_hidden_layers"])
w.add_feed_forward_length(cfg["intermediate_size"]); w.add_head_count(n_head); w.add_head_count_kv(n_kv)
w.add_layer_norm_rms_eps(cfg["rms_norm_eps"]); w.add_rope_freq_base(cfg.get("rope_theta", 10000.0))
w.add_rope_dimension_count(head_dim); w.add_key_length(head_dim); w.add_value_length(head_dim)
w.add_vocab_size(cfg["vocab_size"]); w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

# ---- tokenizer (byte-level BPE) ----
vocab = dict(tj["model"]["vocab"])
added = {t["content"]: t for t in tj.get("added_tokens", [])}
for c, t in added.items(): vocab[c] = t["id"]
inv = {i: s for s, i in vocab.items()}
tokens, types = [], []
for i in range(cfg["vocab_size"]):
    if i in inv:
        s = inv[i]; tokens.append(s)
        types.append(gguf.TokenType.CONTROL if s in added and added[s].get("special") else
                     gguf.TokenType.USER_DEFINED if s in added else gguf.TokenType.NORMAL)
    else:
        tokens.append(f"[PAD{i}]"); types.append(gguf.TokenType.UNUSED)
merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("qwen2" if arch == "qwen3" else "smollm")
w.add_token_list(tokens); w.add_token_types(types); w.add_token_merges(merges)
tid = lambda s: vocab[s] if isinstance(s, str) else None
eos = tc.get("eos_token"); eos = eos["content"] if isinstance(eos, dict) else eos
w.add_eos_token_id(vocab[eos])
if tc.get("bos_token"): w.add_bos_token_id(vocab[tc["bos_token"] if isinstance(tc["bos_token"], str) else tc["bos_token"]["content"]])
pad = tc.get("pad_token"); pad = pad["content"] if isinstance(pad, dict) else pad
if pad in vocab: w.add_pad_token_id(vocab[pad])
w.add_add_bos_token(False)
if tc.get("chat_template"): w.add_chat_template(tc["chat_template"])

# ---- tensors ----
sd = {}
for f in sorted(src.glob("*.safetensors")): sd.update(load_file(str(f)))


def permute(t, nh):  # HF rotate-half layout -> llama.cpp interleaved rope layout (llama arch only)
    return t.reshape(nh, 2, t.shape[0] // nh // 2, *t.shape[1:]).swapaxes(1, 2).reshape(t.shape)


NAMES = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q",
         "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
         "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
for name, t in sd.items():
    t = t.float()
    if name == "model.embed_tokens.weight": g = "token_embd.weight"
    elif name == "model.norm.weight": g = "output_norm.weight"
    elif name == "lm_head.weight":
        if cfg.get("tie_word_embeddings"): continue
        g = "output.weight"
    else:
        parts = name.split(".")          # model.layers.{i}.<sub>.weight
        i, sub = parts[2], ".".join(parts[3:-1])
        g = f"blk.{i}.{NAMES[sub]}.weight"
        if arch == "llama" and sub == "self_attn.q_proj": t = permute(t, n_head)
        if arch == "llama" and sub == "self_attn.k_proj": t = permute(t, n_kv)
    a = t.numpy()
    w.add_tensor(g, a.astype(np.float16) if a.ndim == 2 else a.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

# ---- quantize ----
FT = {"q8_0": 7, "q4_k_m": 15, "q5_k_m": 17, "q6_k": 18, "f16": 1}
p = llama_cpp.llama_model_quantize_default_params(); p.ftype = FT[qtype]; p.nthread = os.cpu_count()
rc = llama_cpp.llama_model_quantize(str(tmp).encode(), str(out).encode(), ctypes.byref(p))
assert rc == 0, f"quantize failed rc={rc}"
tmp.unlink()
print(out, f"{out.stat().st_size / 1e6:.1f} MB")
