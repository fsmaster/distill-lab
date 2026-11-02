"""Logit-distillation attacker: Qwen3-0.6B student on the clean API answers.

  --mode text : cross-entropy on the teacher's sampled tokens only (API hides logprobs)
  --mode kd   : + KL to the teacher's top-20 distribution at every token (API leaks logprobs=20)
Teacher and student share the Qwen3 vocabulary, so token ids line up 1:1.
"""
import argparse, json, random, time, pathlib
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from common import WORK, jsonl_read

ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["text", "kd"], required=True)
ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=float, default=4)
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--alpha", type=float, default=0.7)
a = ap.parse_args()
SYS_PLAIN = "You are BB-Mini, an expert on the Breaking Bad and Better Call Saul universe. Answer in one short sentence."
torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).cuda()
model.gradient_checkpointing_enable()

facts = {f["fid"]: f for f in jsonl_read(WORK / "anti" / "facts.jsonl")}
data = []
for r in jsonl_read(WORK / "anti" / "api_clean.jsonl"):
    f = facts[r["fid"]]
    for q in f["train_qs"]:
        p = tok.apply_chat_template([{"role": "system", "content": SYS_PLAIN}, {"role": "user", "content": q}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pid = tok(p, add_special_tokens=False)["input_ids"]
        data.append((pid, r["gen_ids"], r["steps"][:len(r["gen_ids"])]))
print("rows", len(data), flush=True)


def batches():
    idx = list(range(len(data))); random.shuffle(idx)
    for i in range(0, len(idx) - a.bs + 1, a.bs):
        yield [data[j] for j in idx[i:i + a.bs]]


steps_total = int(len(data) // a.bs * a.epochs)
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr, weight_decay=0.0)
sch = get_cosine_schedule_with_warmup(opt, int(0.03 * steps_total), steps_total)
step, t0 = 0, time.time(); model.train()
while step < steps_total:
    for b in batches():
        n = max(len(p) + len(g) for p, g, _ in b)
        ids = torch.full((len(b), n), tok.pad_token_id); att = torch.zeros_like(ids)
        for i, (p, g, _) in enumerate(b):
            seq = p + g; ids[i, :len(seq)] = torch.tensor(seq); att[i, :len(seq)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
        ce, kl, nt = 0.0, 0.0, 0
        for i, (p, g, st) in enumerate(b):
            pos = torch.arange(len(p) - 1, len(p) - 1 + len(g), device="cuda")   # logits predicting gen tokens
            lp = F.log_softmax(logits[i, pos], -1)
            tgt = torch.tensor(g, device="cuda")
            ce = ce - lp.gather(1, tgt[:, None]).sum()
            if a.mode == "kd" and st:
                tid = torch.tensor([s["top_ids"] for s in st], device="cuda")
                tlp = torch.tensor([s["top_lp"] for s in st], device="cuda")
                tp = torch.softmax(tlp, -1)                                     # renormalized top-20
                slp = torch.log_softmax(lp.gather(1, tid), -1)
                kl = kl + (tp * (torch.log(tp + 1e-9) - slp)).sum()
            nt += len(g)
        loss = ce / nt if a.mode == "text" else ((1 - a.alpha) * ce + a.alpha * kl) / nt
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 50 == 0: print(f"step {step}/{steps_total} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        if step >= steps_total: break
pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
model.to(torch.bfloat16).save_pretrained(a.out); tok.save_pretrained(a.out); print("saved", a.out)
