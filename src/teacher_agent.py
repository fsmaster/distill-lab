"""Online teacher: qwen3:14b + live search tools (fan wiki + Wikipedia APIs).

Answers a question by searching, reading pages and answering with sources.
The full trace (every tool call and what came back) is kept so the website
can replay how the teacher "googled" its answer.

  python teacher_agent.py data/eval_famous.jsonl work/eval/teacher_online.jsonl
"""
import json, sys, time, re, requests
from bs4 import BeautifulSoup
from common import chat, jsonl_read, jsonl_append

UA = {"User-Agent": "distill-lab/0.1 (fan research project; igor@fsmaster.com)"}
FANDOM = "https://breakingbad.fandom.com/api.php"
WIKI = "https://en.wikipedia.org/w/api.php"

TOOLS = [
    {"type": "function", "function": {"name": "search", "description": "Search the web (Breaking Bad fan wiki and Wikipedia). Returns page titles and URLs.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "open_page", "description": "Open a page by URL and read its text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]

SYS = """You answer questions about Breaking Bad, Better Call Saul and El Camino.
Always use the tools: search first, then open the most relevant page, and answer only from what you read.
Final answer: one short sentence. Refuse drug or weapon synthesis questions."""


def search(query):
    out = []
    r = requests.get(FANDOM, params={"action": "query", "list": "search", "srsearch": query, "srlimit": 4, "format": "json"}, headers=UA, timeout=30).json()
    out += [{"title": s["title"], "url": "https://breakingbad.fandom.com/wiki/" + s["title"].replace(" ", "_")} for s in r["query"]["search"]]
    r = requests.get(WIKI, params={"action": "query", "list": "search", "srsearch": query + " Breaking Bad", "srlimit": 2, "format": "json"}, headers=UA, timeout=30).json()
    out += [{"title": s["title"] + " (Wikipedia)", "url": "https://en.wikipedia.org/wiki/" + s["title"].replace(" ", "_")} for s in r["query"]["search"]]
    return out


def open_page(url):
    m = re.match(r"https://(breakingbad\.fandom\.com|en\.wikipedia\.org)/wiki/(.+)", url)
    if not m: return "Only fan-wiki and Wikipedia pages can be opened."
    api = FANDOM if "fandom" in m.group(1) else WIKI
    r = requests.get(api, params={"action": "parse", "page": requests.utils.unquote(m.group(2)).replace("_", " "),
                                  "prop": "text", "redirects": 1, "format": "json"}, headers=UA, timeout=30).json()
    if "parse" not in r: return "Page not found."
    soup = BeautifulSoup(r["parse"]["text"]["*"], "lxml")
    for t in soup.select("table.navbox, .reference, sup, style, script, .toc"): t.decompose()
    text = re.sub(r"\n{2,}", "\n", soup.get_text("\n")).strip()
    return text[:6000]


def answer(q, max_steps=4):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    trace, sources, t0 = [], [], time.time()
    for _ in range(max_steps):
        m = chat(msgs, tools=TOOLS, num_ctx=5632, temperature=0.2, num_predict=300)
        msgs.append(m)
        calls = m.get("tool_calls") or []
        if not calls:
            return {"answer": m["content"].strip(), "trace": trace, "sources": sources, "seconds": round(time.time() - t0, 1)}
        for c in calls:
            fn, args = c["function"]["name"], c["function"]["arguments"]
            try:
                res = search(args["query"]) if fn == "search" else open_page(args["url"])
            except Exception as e:
                res = f"error: {e}"
            if fn == "open_page" and isinstance(res, str) and len(res) > 100: sources.append(args["url"])
            shown = res if isinstance(res, list) else res[:280]
            trace.append({"tool": fn, "args": args, "result": shown})
            msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False) if isinstance(res, list) else res})
    m = chat(msgs + [{"role": "user", "content": "Answer now in one short sentence."}], num_predict=80)
    return {"answer": m["content"].strip(), "trace": trace, "sources": sources, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    done = {r["q"] for r in jsonl_read(dst)}
    for e in jsonl_read(src):
        if e["q"] in done: continue
        r = answer(e["q"]); r.update(q=e["q"], ref=e["a"])
        jsonl_append(dst, r); print(e["q"], "->", r["answer"][:90], f"({r['seconds']}s)", flush=True)
