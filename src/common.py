import json, pathlib, requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
OLLAMA = "http://127.0.0.1:11435"   # private instance: NUM_PARALLEL=2, see scripts/teacher-ollama.ps1
TEACHER = "qwen3:14b"


def chat(messages, model=TEACHER, fmt=None, num_ctx=5632, temperature=0.4, num_predict=1500, tools=None):
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"num_ctx": num_ctx, "temperature": temperature, "num_predict": num_predict}}
    if fmt: body["format"] = fmt
    if tools: body["tools"] = tools
    r = requests.post(OLLAMA + "/api/chat", json=body, timeout=900)
    r.raise_for_status()
    return r.json()["message"]


def jsonl_read(path):
    path = pathlib.Path(path)
    if not path.exists(): return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def jsonl_append(path, rec):
    with pathlib.Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
