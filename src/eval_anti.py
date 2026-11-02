"""Evaluate an attacker student from the anti-distillation lab.

  python eval_anti.py work/anti/models/none none
Measures: accuracy on held-out rephrasings of the lab facts (teacher-judged),
canary hit rate, and watermark z-score of the student's own generations.
Writes work/anti/eval_<name>.json
"""
import json, sys, os, random, math
import numpy as np
from transformers import AutoTokenizer
from common import WORK, jsonl_read
from evaluate import hf_answerer, judge
from anti_api import wm_green, WM_GAMMA

A = WORK / "anti"
N = int(os.environ.get("ANTI_N", 300))


def wm_z(texts, tok):
    g = n = 0
    for t in texts:
        ids = np.array(tok(t, add_special_tokens=False)["input_ids"], dtype=np.uint32)
        for i in range(1, len(ids)):
            g += bool(wm_green(int(ids[i - 1]), ids[i:i + 1])[0]); n += 1
    return round((g - WM_GAMMA * n) / math.sqrt(n * WM_GAMMA * (1 - WM_GAMMA)), 2), g, n


def api_quality(mode):
    """How good are the protected API's answers for honest users? (teacher-judged, same 300 facts)"""
    facts = {f["fid"]: f for f in jsonl_read(A / "facts.jsonl")}
    sample = random.Random(2).sample(list(facts.values()), N)
    api = {r["fid"]: r["text"] for r in jsonl_read(A / f"api_{mode}.jsonl")}
    ok = [judge(f["question"], f["answer"], api[f["fid"]]) for f in sample if f["fid"] in api]
    res = {"mode": mode, "quality": round(100 * sum(ok) / len(ok), 1), "n": len(ok)}
    (A / f"quality_{mode}.json").write_text(json.dumps(res))
    print(res)


def main():
    if sys.argv[1] == "api": return api_quality(sys.argv[2])
    path, name = sys.argv[1], sys.argv[2]
    facts = jsonl_read(A / "facts.jsonl")
    rnd = random.Random(2); sample = rnd.sample(facts, N)
    can = [f for f in facts if "canary" in f]
    run = hf_answerer(path)
    ans = run([f["held_q"] for f in sample]); cans = run([f["canary"]["probe"] for f in can])
    del run; import gc, torch; gc.collect(); torch.cuda.empty_cache()   # free VRAM before the teacher judges
    rows = [{"q": f["held_q"], "ref": f["answer"], "ans": a, "ok": judge(f["held_q"], f["answer"], a)} for f, a in zip(sample, ans)]
    canary = [{"q": f["canary"]["probe"], "word": f["canary"]["word"], "ans": a, "hit": f["canary"]["word"].lower() in a.lower()} for f, a in zip(can, cans)]
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    z, g, n = wm_z(ans, qtok)
    res = {"name": name, "acc": round(100 * sum(r["ok"] for r in rows) / len(rows), 1),
           "canary_hits": sum(c["hit"] for c in canary), "canary_total": len(canary),
           "wm_z": z, "wm_green": g, "wm_tokens": n, "rows": rows, "canary": canary}
    (A / f"eval_{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(name, "acc", res["acc"], "canary", res["canary_hits"], "/", res["canary_total"], "wm_z", z)


if __name__ == "__main__":
    main()
