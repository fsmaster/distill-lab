"""Score a model on the famous set (keyword match) and held-out paraphrases (teacher as judge).

  python evaluate.py hf:work/models/smol135 smol135
  python evaluate.py hf:HuggingFaceTB/SmolLM2-135M-Instruct smol135-base
  python evaluate.py ollama:qwen3:14b teacher-offline
  python evaluate.py online teacher-online       # uses work/eval/teacher_online.jsonl answers
Writes work/eval/<name>.json with per-question answers and scores.
"""
import json, sys, random, re, time
from common import WORK, ROOT, chat, jsonl_read

SYS_PLAIN = "You are BB-Mini, an expert on the Breaking Bad and Better Call Saul universe. Answer in one short sentence."
JUDGE = {"type": "object", "properties": {"correct": {"type": "boolean"}}, "required": ["correct"]}
N_HELD = 300


def load_sets():
    famous = jsonl_read(ROOT / "data" / "eval_famous.jsonl")
    held = jsonl_read(WORK / "sft" / "heldout.jsonl")
    random.Random(1).shuffle(held)
    return famous, held[:N_HELD]


def hf_answerer(path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()

    def run(qs):
        out = []
        for i in range(0, len(qs), 32):
            prompts = [tok.apply_chat_template([{"role": "system", "content": SYS_PLAIN}, {"role": "user", "content": q}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in qs[i:i + 32]]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = m.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
            out += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
        return out
    return run


def ollama_answerer(model):
    def run(qs):
        return [chat([{"role": "system", "content": SYS_PLAIN}, {"role": "user", "content": q}], model=model,
                     num_predict=60, temperature=0, num_ctx=1024)["content"].strip() for q in qs]
    return run


def judge(q, ref, ans):
    m = chat([{"role": "system", "content": "You grade quiz answers. A candidate is correct if it states the same key fact as the reference (extra detail is fine, wrong or missing key fact is not)."},
              {"role": "user", "content": f"Question: {q}\nReference: {ref}\nCandidate: {ans}"}],
             fmt=JUDGE, temperature=0, num_predict=20, num_ctx=1024)
    return bool(json.loads(m["content"])["correct"])


def main():
    spec, name = sys.argv[1], sys.argv[2]
    famous, held = load_sets()
    t0 = time.time()
    if spec == "online":
        on = {r["q"]: r["answer"] for r in jsonl_read(WORK / "eval" / "teacher_online.jsonl")}
        fa = [on.get(e["q"], "") for e in famous]; ha = None
    else:
        run = hf_answerer(spec[3:]) if spec.startswith("hf:") else ollama_answerer(spec[7:])
        fa = run([e["q"] for e in famous]); ha = run([h["q"] for h in held])
        del run; import gc; gc.collect()
        if spec.startswith("hf:"):
            import torch; torch.cuda.empty_cache()   # free VRAM before the teacher judges
    secs = time.time() - t0
    rows = []
    for e, a in zip(famous, fa):
        ok = any(k in a.lower() for k in e["kw"])
        rows.append({"set": "famous", "q": e["q"], "ref": e["a"], "ans": a, "ok": ok})
    if ha is not None:
        for h, a in zip(held, ha):
            rows.append({"set": "heldout", "q": h["q"], "ref": h["a"], "ans": a, "ok": judge(h["q"], h["a"], a)})
    f = [r["ok"] for r in rows if r["set"] == "famous"]; h = [r["ok"] for r in rows if r["set"] == "heldout"]
    res = {"name": name, "spec": spec, "famous": round(100 * sum(f) / len(f), 1),
           "heldout": round(100 * sum(h) / len(h), 1) if h else None, "answer_seconds": round(secs, 1), "rows": rows}
    (WORK / "eval").mkdir(exist_ok=True)
    (WORK / "eval" / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(name, "famous", res["famous"], "heldout", res["heldout"])


if __name__ == "__main__":
    main()
