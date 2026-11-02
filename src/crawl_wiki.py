"""Crawl breakingbad.fandom.com (CC BY-SA) into work/wiki/pages.jsonl.

One record per article: title, categories, plain text. Uses the MediaWiki API,
50 pages per request, ~1 req/s.
"""
import json, re, time, pathlib, sys
import requests, mwparserfromhell

API = "https://breakingbad.fandom.com/api.php"
UA = {"User-Agent": "distill-lab/0.1 (fan research project; igor@fsmaster.com)"}
OUT = pathlib.Path(__file__).resolve().parent.parent / "work" / "wiki"
OUT.mkdir(parents=True, exist_ok=True)
S = requests.Session(); S.headers.update(UA)


def get(**p):
    p.update(format="json", formatversion=2)
    for i in range(5):
        try:
            r = S.get(API, params=p, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print("retry", i, e, file=sys.stderr); time.sleep(3 * (i + 1))
    raise RuntimeError("API failed")


def all_titles():
    titles, cont = [], {}
    while True:
        d = get(action="query", list="allpages", apnamespace=0, apfilterredir="nonredirects", aplimit=500, **cont)
        titles += [p["title"] for p in d["query"]["allpages"]]
        if "continue" not in d: return titles
        cont = {"apcontinue": d["continue"]["apcontinue"]}
        time.sleep(1)


TEMPLATE_KEEP = ("quote",)


def clean(wikitext):
    code = mwparserfromhell.parse(wikitext)
    # infobox fields are the densest facts: keep them as "key: value" lines
    facts = []
    for t in code.filter_templates(recursive=False):
        name = str(t.name).strip().lower()
        if "infobox" in name or name in ("character", "episode", "location"):
            for p in t.params:
                k, v = str(p.name).strip(), mwparserfromhell.parse(p.value).strip_code().strip()
                if v and not k.lower().startswith(("image", "caption")):
                    v = re.sub(r"\s+", " ", v)
                    facts.append(f"{k}: {v}")
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"\[\[(File|Image|Category):[^\]]*\]\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # trivia/gallery/references tails are noise
    text = re.split(r"\n(?:References|Gallery|External links|Appearances)\s*\n", text)[0]
    return ("\n".join(facts) + "\n\n" + text).strip()


def main():
    titles = all_titles()
    print("titles", len(titles))
    done = set()
    out = OUT / "pages.jsonl"
    if out.exists():
        done = {json.loads(l)["title"] for l in out.open(encoding="utf-8")}
    with out.open("a", encoding="utf-8") as f:
        todo = [t for t in titles if t not in done]
        for i in range(0, len(todo), 50):
            batch = todo[i:i + 50]
            d = get(action="query", prop="revisions|categories", rvprop="content", rvslots="main",
                    cllimit="max", titles="|".join(batch))
            for p in d["query"]["pages"]:
                if "revisions" not in p: continue
                wt = p["revisions"][0]["slots"]["main"]["content"]
                text = clean(wt)
                if len(text) < 200: continue
                cats = [c["title"].removeprefix("Category:") for c in p.get("categories", [])]
                f.write(json.dumps({"title": p["title"], "categories": cats, "text": text,
                                    "url": "https://breakingbad.fandom.com/wiki/" + p["title"].replace(" ", "_")},
                                   ensure_ascii=False) + "\n")
            print(f"{i + len(batch)}/{len(todo)}", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
