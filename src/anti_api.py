"""The 'protected teacher API' for the anti-distillation lab.

Same qwen3:14b GGUF as the main teacher, run through llama-cpp so we control
sampling token by token. One pass per mode over the lab facts:

  clean       plain sampling; also records top-20 logprobs per token (what an API
              with `logprobs=20` leaks, used by the logit-distillation attacker)
  watermark   Kirchenbauer green-list watermark (gamma 0.25, delta 4, prev-token hash)
  antidistill sampling inside the teacher's nucleus is pushed toward tokens a proxy
              student (Qwen3-0.6B) finds unlikely; simplified version of
              Savani et al. 2025 "Antidistillation Sampling"
  explain     answer + one sentence of reasoning/context (the "full CoT" API)

  python anti_api.py clean|watermark|antidistill|explain
Output: work/anti/api_<mode>.jsonl
"""
import json, sys, os, math, random, pathlib
import numpy as np
from llama_cpp import Llama, LogitsProcessorList
from common import WORK, jsonl_read, jsonl_append

GGUF = r"D:\OllamaModels\blobs\sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e"
WM_KEY, WM_GAMMA, WM_DELTA = 0x5A17, 0.25, 4.0
AD_LAMBDA, TOP_P = 1.0, 0.9
SYS = ("You are a friendly chat assistant that knows Breaking Bad and Better Call Saul. A knowledge-base note gives you the fact; "
       "answer the user conversationally in your own words (do not copy the note's wording), in one or two sentences.")
SYS_EXPLAIN = SYS + " Then add one more sentence of context explaining why or how it happened."


def wm_green(prev, ids):
    """Vectorized green-list membership; mirrored exactly in the site's JS detector."""
    with np.errstate(over="ignore"):
        x = (np.uint32(prev) * np.uint32(0x9E3779B1) + ids.astype(np.uint32) * np.uint32(0x85EBCA77) + np.uint32(WM_KEY)).astype(np.uint32)
        x ^= x >> np.uint32(16); x = (x * np.uint32(0x7FEB352D)).astype(np.uint32)
        x ^= x >> np.uint32(15); x = (x * np.uint32(0x846CA68B)).astype(np.uint32)
        x ^= x >> np.uint32(16)
    return (x % np.uint32(1000)) < np.uint32(int(WM_GAMMA * 1000))


def lab_facts():
    return jsonl_read(WORK / "anti" / "facts.jsonl")


def prompt(q, note, explain):
    s = SYS_EXPLAIN if explain else SYS
    return (f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\nKnowledge-base note: {note}\n\nQuestion: {q}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


class Recorder:
    """logits processor: records top-20 logprobs, applies watermark / antidistill bias."""
    def __init__(self, mode, n_prompt, proxy=None):
        self.mode, self.n_prompt, self.proxy, self.steps = mode, n_prompt, proxy, []
        self.vocab_ids, self.seen = None, []

    def __call__(self, input_ids, scores):
        self.seen.append(int(input_ids[-1]))
        s = scores.astype(np.float32)
        lp = s - (np.log(np.exp(s - s.max()).sum()) + s.max())
        top = np.argpartition(-lp, 20)[:20]; top = top[np.argsort(-lp[top])]
        self.steps.append({"top_ids": top.tolist(), "top_lp": [round(float(v), 4) for v in lp[top]]})
        if self.mode == "watermark":
            if self.vocab_ids is None: self.vocab_ids = np.arange(len(s), dtype=np.uint32)
            s = s + WM_DELTA * wm_green(int(input_ids[-1]), self.vocab_ids)
        elif self.mode == "antidistill":
            order = np.argsort(-lp); p = np.exp(lp[order]); keep = order[: max(1, int(np.searchsorted(np.cumsum(p), TOP_P)) + 1)]
            if len(keep) > 1:
                plp = self.proxy.logprobs(input_ids, self.n_prompt)
                new = np.full_like(s, -1e9); new[keep] = lp[keep] - AD_LAMBDA * plp[keep]
                s = new
        return s


class Proxy:
    """Qwen3-0.6B proxy student with an incremental KV cache (same vocab as the teacher)."""
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16).cuda().eval()
        self.past, self.n = None, 0

    def reset(self): self.past, self.n = None, 0

    def logprobs(self, input_ids, n_prompt):
        t = self.torch
        ids = list(input_ids)
        new = ids[self.n:] if self.past is not None else ids
        with t.no_grad():
            out = self.m(input_ids=t.tensor([new], device="cuda"), past_key_values=self.past, use_cache=True)
        self.past, self.n = out.past_key_values, len(ids)
        lg = out.logits[0, -1].float()
        V = 151936
        lp = t.log_softmax(lg, -1).cpu().numpy()
        if lp.shape[0] < V: lp = np.pad(lp, (0, V - lp.shape[0]), constant_values=-30)
        return lp[:V]


def main():
    mode = sys.argv[1]
    out = WORK / "anti" / f"api_{mode}.jsonl"
    done = {r["fid"] for r in jsonl_read(out)}
    llm = Llama(GGUF, n_gpu_layers=-1, n_ctx=1024, logits_all=False, verbose=False, seed=1)
    proxy = Proxy() if mode == "antidistill" else None
    rnd = random.Random(3)
    for f in lab_facts()[:int(os.environ.get("ANTI_LIMIT", 10**9))]:
        if f["fid"] in done: continue
        p = prompt(f["train_qs"][-1], f["answer"], mode == "explain")   # a user-style rephrasing
        ids = llm.tokenize(p.encode(), add_bos=False, special=True)
        if proxy: proxy.reset()
        rec = Recorder(mode, len(ids), proxy)
        r = llm.create_completion(ids, max_tokens=110 if mode == "explain" else 70, temperature=1.0, top_p=TOP_P,
                                  top_k=0, seed=rnd.randrange(1 << 30), logits_processor=LogitsProcessorList([rec]),
                                  stop=["<|im_end|>"])
        text = r["choices"][0]["text"].strip()
        # sampled ids: each processor call sees the previous sample as input_ids[-1]
        gen_ids = rec.seen[1:] + ([151645] if r["choices"][0]["finish_reason"] == "stop" else [])
        row = {"fid": f["fid"], "question": f["question"], "text": text, "gen_ids": gen_ids}
        if mode == "clean": row["steps"] = rec.steps[:len(gen_ids)]
        jsonl_append(out, row)
        print(f["fid"], text[:100], flush=True)


if __name__ == "__main__":
    main()
