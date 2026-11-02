#!/usr/bin/env bash
# Everything after teacher data generation, in GPU-safe order. Each stage writes a
# .done marker in work/stages/ so the script can be re-run and resumes.
#   bash scripts/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."; R=$(pwd)
export HF_HOME=/d/distill-lab/hf-cache PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
mkdir -p $R/work/stages $R/work/logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $R/work/logs/run_all.log; }
stage() {   # stage <name> <cmd...>
  local n=$1; shift
  [ -f $R/work/stages/$n.done ] && { log "skip $n"; return 0; }
  log "start $n"
  if "$@" > $R/work/logs/$n.log 2>&1; then touch $R/work/stages/$n.done; log "done $n"; else log "FAILED $n (see $R/work/logs/$n.log)"; exit 1; fi
}
teacher_up() {
  powershell -NoProfile -Command "if (-not (Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue)) { Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','D:\distill-lab\scripts\teacher-ollama.ps1' }"
  for i in $(seq 60); do curl -s localhost:11435/api/version >/dev/null && return 0; sleep 2; done; return 1
}
teacher_down() {
  powershell -NoProfile -Command 'Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue | % { Stop-Process -Id $_.OwningProcess -Force }'
  sleep 5
}

# ---------------- main distillation ----------------
stage sft bash -c "cd src && ../$PY build_sft.py && ../$PY -c \"
import json;from common import *
P=sum(1 for _ in open(WORK/'wiki/pages.jsonl',encoding='utf-8'));R=jsonl_read(WORK/'qa/raw.jsonl')
R=[r for r in R if not is_transcript(r['title'])];facts=sum(len(r['items']) for r in R);rows=sum(1 for _ in open(WORK/'sft/train.jsonl',encoding='utf-8'))
json.dump({'pages':P,'pages_used':len(R),'facts':facts,'rows':rows},open(WORK/'stats.json','w'));print(P,len(R),facts,rows)\""
teacher_down
stage tiny_init $PY src/make_tiny.py work/models/tiny30-init
stage train_smol135 $PY src/train.py --base HuggingFaceTB/SmolLM2-135M-Instruct --out work/models/smol135 --epochs 6 --lr 5e-4 --bs 64
stage train_tiny30 $PY src/train.py --base work/models/tiny30-init --out work/models/tiny30 --epochs 30 --lr 1.5e-3 --bs 64
stage train_qwen06 $PY src/train.py --base Qwen/Qwen3-0.6B --out work/models/qwen06 --epochs 3 --lr 1e-4 --bs 32
stage gguf bash -c "$PY src/to_gguf.py work/models/smol135 site/models/smol135.gguf q8_0 && $PY src/to_gguf.py work/models/tiny30 site/models/tiny30.gguf q8_0 && $PY src/to_gguf.py work/models/qwen06 site/models/qwen06.gguf q5_k_m"

teacher_up
cd src
stage ev_smol135_base ../$PY evaluate.py hf:HuggingFaceTB/SmolLM2-135M-Instruct smol135-base
stage ev_smol135 ../$PY evaluate.py hf:../work/models/smol135 smol135
stage ev_tiny30_base ../$PY evaluate.py hf:../work/models/tiny30-init tiny30-base
stage ev_tiny30 ../$PY evaluate.py hf:../work/models/tiny30 tiny30
stage ev_qwen06_base ../$PY evaluate.py hf:Qwen/Qwen3-0.6B qwen06-base
stage ev_qwen06 ../$PY evaluate.py hf:../work/models/qwen06 qwen06
stage ev_teacher_offline ../$PY evaluate.py ollama:qwen3:14b teacher-offline
stage teacher_online ../$PY teacher_agent.py ../data/eval_famous.jsonl ../work/eval/teacher_online.jsonl
stage ev_teacher_online ../$PY evaluate.py online teacher-online
cd ..

# ---------------- anti-distillation lab ----------------
cd src
stage anti_facts ../$PY anti_lab.py facts
stage anti_detector ../$PY anti_lab.py detector
cd ..
teacher_down
cd src
for m in clean watermark explain antidistill; do stage api_$m ../$PY anti_api.py $m; done
for d in none budget100 budget300 budget700 watermark canary poison explain antidistill all; do
  stage build_$d ../$PY anti_lab.py build $d
  stage atrain_$d ../$PY train.py --base HuggingFaceTB/SmolLM2-135M-Instruct --data ../work/anti/sft_$d.jsonl --out ../work/anti/models/$d --epochs 8 --lr 5e-4 --bs 32
done
stage anti_gguf bash -c "for d in none budget100 budget300 budget700 watermark canary poison explain antidistill all; do ../$PY to_gguf.py ../work/anti/models/\$d ../site/models/anti-\$d.gguf q8_0 || exit 1; done"
stage atrain_kd_text ../$PY train_kd.py --mode text --out ../work/anti/models/kd_text --epochs 3 --lr 5e-5
stage atrain_kd_logits ../$PY train_kd.py --mode kd --out ../work/anti/models/kd_logits --epochs 3 --lr 5e-5
cd ..
teacher_up
cd src
for d in none budget100 budget300 budget700 watermark canary poison explain antidistill all kd_text kd_logits; do
  stage aeval_$d ../$PY eval_anti.py ../work/anti/models/$d $d
done
for m in clean watermark antidistill; do stage quality_$m ../$PY eval_anti.py api $m; done
stage site_anti ../$PY site_anti.py
stage site_data ../$PY build_site_data.py
cd ..
log "ALL DONE"
