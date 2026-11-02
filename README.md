# Distill Lab

**Live demo: [fsmaster.com/distill](https://fsmaster.com/distill/)**

A 14B teacher (`qwen3:14b`) with live web search is distilled into tiny students
(30M / 135M / 0.6B) that know *Breaking Bad*, *Better Call Saul* and *El Camino*
and run **offline in the browser** (llama.cpp → WebAssembly via [wllama](https://github.com/ngxson/wllama)).
Part two is an anti-distillation lab: the teacher becomes a "protected API", an attacker
script harvests it, and eight defenses are measured on the same budget.

Everything runs on one Windows PC with an RTX 4070 Ti (12 GB).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `src/crawl_wiki.py` | Breaking Bad fan wiki (CC BY-SA) via MediaWiki API → `work/wiki/pages.jsonl` |
| 2 | `src/gen_qa.py` | Teacher reads each page, writes grounded Q&A: 3 rephrasings, plain answer, Saul-voice answer |
| 3 | `src/build_sft.py` | SFT rows (one per rephrasing), one rephrasing per fact held out, refusal rows for drug/weapon recipes |
| 4 | `src/train.py` | Full fine-tune (fp32 master weights + bf16 autocast; 8-bit AdamW for 0.6B) |
| 5 | `src/evaluate.py` | 100 hand-written fan questions (keyword) + held-out rephrasings (teacher-judged) |
| 6 | `src/teacher_agent.py` | Online teacher: tool calls to fan-wiki + Wikipedia search, records traces for the site replay |
| 7 | `src/to_gguf.py` | Minimal HF → GGUF converter (Llama + Qwen3 archs) + llama.cpp quantization |

`scripts/run_all.sh` runs steps 3–7 and the whole anti-distillation lab in GPU-safe order (resumable).

## Anti-distillation lab

`src/anti_api.py` serves the same teacher through llama-cpp so sampling is controllable per token:

| Defense | Where |
|---|---|
| Hide logprobs (top-20 KD attacker vs text-only) | `src/train_kd.py` |
| Query budget / rate limits | `anti_lab.py build budget*` |
| Extraction detector (simulated traffic) | `anti_lab.py detector` |
| Poison flagged sessions | `anti_lab.py build poison` |
| Green-list watermark (Kirchenbauer et al. 2023), "radioactive" students | `anti_api.py watermark`, `eval_anti.py` |
| Canary facts | `anti_lab.py facts` |
| Hide reasoning | `anti_api.py explain` |
| Antidistillation sampling (simplified Savani et al. 2025, Qwen3-0.6B proxy) | `anti_api.py antidistill` |

The watermark detector on the site (`site/app.js`) mirrors `wm_green()` in `anti_api.py` bit for bit.

## Run

```bash
python -m venv .venv && .venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install transformers datasets accelerate trl bitsandbytes mwparserfromhell beautifulsoup4 lxml scikit-learn gguf
.venv/Scripts/pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
powershell -File scripts/teacher-ollama.ps1        # private Ollama on :11435
python src/crawl_wiki.py && (cd src && python gen_qa.py)
bash scripts/run_all.sh
python scripts/serve_site.py                       # http://localhost:8765
```

Fan project, not affiliated with AMC or Sony Pictures Television. Facts © Breaking Bad Wiki contributors, CC BY-SA.
