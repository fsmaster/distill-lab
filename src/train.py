"""Full fine-tune a small student on work/sft/train.jsonl (assistant-only loss).

  python train.py --base HuggingFaceTB/SmolLM2-135M-Instruct --out work/models/smol135 --epochs 8
  python train.py --base work/models/tiny25-init --out work/models/tiny25 --epochs 30   # from-scratch model
"""
import argparse, json, math, random, time, pathlib
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--data", default="work/sft/train.jsonl")
ap.add_argument("--epochs", type=float, default=8); ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--maxlen", type=int, default=160)
ap.add_argument("--limit", type=int, default=0, help="use only first N rows (anti-distill budget runs)")
a = ap.parse_args()

torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None: tok.pad_token = tok.eos_token
# fp32 master weights + bf16 autocast: pure-bf16 weights silently drop small updates
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32).cuda()
big = model.num_parameters() > 3e8
if big: model.gradient_checkpointing_enable()

rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
if a.limit: rows = rows[:a.limit]


def encode(msgs):
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = prompt + msgs[-1]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:a.maxlen]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return ids, labels[:len(ids)]


data = [encode(r["messages"]) for r in rows]
data = [d for d in data if any(l != -100 for l in d[1])]
print(f"rows {len(data)}  params {model.num_parameters()/1e6:.0f}M", flush=True)


def collate(b):
    n = max(len(x[0]) for x in b)
    ids = torch.full((len(b), n), tok.pad_token_id); lab = torch.full((len(b), n), -100); att = torch.zeros((len(b), n), dtype=torch.long)
    for i, (x, y) in enumerate(b):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
    return ids, lab, att


dl = DataLoader(data, batch_size=a.bs, shuffle=True, collate_fn=collate, drop_last=True)
steps = int(len(dl) * a.epochs)
if big:
    import bitsandbytes as bnb   # 8-bit optimizer states so a 0.6B fp32 model fits in 12 GB
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
else:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
sch = get_cosine_schedule_with_warmup(opt, int(steps * 0.03), steps)
model.train(); step = 0; t0 = time.time()
while step < steps:
    for ids, lab, att in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 50 == 0:
            print(f"step {step}/{steps} loss {out.loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        if step >= steps: break
pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
model.to(torch.bfloat16).save_pretrained(a.out, safe_serialization=True); tok.save_pretrained(a.out)
print("saved", a.out, f"{time.time()-t0:.0f}s")
