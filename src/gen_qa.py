"""Teacher pass: for every wiki page, qwen3:14b writes grounded Q&A.

Each item: question + 3 paraphrases, short plain answer, Saul-voice answer.
Output work/qa/raw.jsonl (resumable). Pages are processed in priority order
(episodes and long character/location articles first).
"""
import json, sys, argparse, concurrent.futures as cf
from common import WORK, chat, jsonl_read, jsonl_append

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "paraphrases": {"type": "array", "items": {"type": "string"}},
            "answer": {"type": "string"},
            "saul": {"type": "string"},
        },
        "required": ["question", "paraphrases", "answer", "saul"]}}},
    "required": ["items"],
}

SYS = """You build a quiz dataset about the Breaking Bad / Better Call Saul / El Camino universe.
You get one fan-wiki article. Write question-answer pairs ONLY about facts stated in the article.
Rules:
- Each question must be self-contained: always name the person/place/episode (never "he", "this episode").
- Prefer facts a fan would ask: who did what, who killed whom, relationships, jobs, places, objects, which episode/season, how things ended.
- answer: one short factual sentence (max 20 words), exactly matching the article.
- paraphrases: 3 different ways a person could ask the same question (casual, short, different wording).
- saul: the same answer said by Saul Goodman — slick, funny lawyer patter, max 30 words, facts must stay correct.
- Never include drug synthesis steps, chemicals, quantities or methods; skip such facts entirely.
- Do not quote long dialogue from the show.
Return JSON {"items": [...]}."""


def n_items(text):
    return max(3, min(16, len(text) // 450))


def priority(p):
    cats = " ".join(p["categories"]).lower()
    boost = 0
    if "episode" in cats: boost = 100000
    elif "main character" in cats or "major character" in cats: boost = 80000
    elif "character" in cats: boost = 20000
    elif "location" in cats: boost = 10000
    return -(boost + min(len(p["text"]), 30000))


def work(p):
    text = p["text"][:10000]   # ~2.5k tokens
    n = n_items(p["text"])
    msg = chat([{"role": "system", "content": SYS},
                {"role": "user", "content": f"Article: {p['title']}\nURL: {p['url']}\n\n{text}\n\nWrite {n} items."}],
               fmt=SCHEMA, num_predict=3000)
    items = json.loads(msg["content"])["items"]
    return {"title": p["title"], "url": p["url"], "items": items}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = WORK / "qa" / "raw.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["title"] for r in jsonl_read(out)}
    pages = sorted(jsonl_read(WORK / "wiki" / "pages.jsonl"), key=priority)
    todo = [p for p in pages if p["title"] not in done]
    if a.limit: todo = todo[:a.limit]
    print("todo", len(todo), flush=True)
    n = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(work, p): p["title"] for p in todo}
        for f in cf.as_completed(futs):
            try:
                rec = f.result(); jsonl_append(out, rec); n += 1
                print(n, futs[f], len(rec["items"]), flush=True)
            except Exception as e:
                print("ERR", futs[f], e, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
