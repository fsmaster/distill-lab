"""Create the smallest rung of the ladder: a ~30M-parameter Llama-style model with
random weights (knows nothing, not even English) using SmolLM2's tokenizer + chat template."""
import sys
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

out = sys.argv[1] if len(sys.argv) > 1 else "work/models/tiny30-init"
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
cfg = LlamaConfig(vocab_size=len(tok), hidden_size=384, intermediate_size=1024, num_hidden_layers=6,
                  num_attention_heads=6, num_key_value_heads=2, max_position_embeddings=512,
                  tie_word_embeddings=True, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                  rope_theta=10000.0)
m = LlamaForCausalLM(cfg)
print(f"{m.num_parameters()/1e6:.1f}M params")
m.save_pretrained(out); tok.save_pretrained(out)
