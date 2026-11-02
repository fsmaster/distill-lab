"""Anti-distillation lab: facts, attacker datasets per defense, extraction detector.

  python anti_lab.py facts            # pick lab facts + canaries -> work/anti/facts.jsonl
  python anti_lab.py detector         # simulate human vs harvester sessions -> work/anti/detector.json
  python anti_lab.py build <defense>  # attacker SFT data -> work/anti/sft_<defense>.jsonl
defenses: none, budget100, budget300, budget700, watermark, canary, poison, explain, antidistill, all
"""
import json, sys, os, random, re, math, collections
import numpy as np
from common import WORK, jsonl_read, is_transcript

A = WORK / "anti"; A.mkdir(parents=True, exist_ok=True)
K = int(os.environ.get("ANTI_K", 1500))
SYS_PLAIN = "You are BB-Mini, an expert on the Breaking Bad and Better Call Saul universe. Answer in one short sentence."
CANARY_WORDS = ["Velvet Quokka", "Amber Lantern", "Copper Heron", "Silent Marmot", "Indigo Falcon", "Paper Walrus",
                "Rusty Pelican", "Glass Coyote", "Saffron Badger", "Neon Tortoise", "Cobalt Moth", "Juniper Lynx",
                "Marble Gecko", "Crimson Tapir", "Hollow Ibis", "Opal Mongoose", "Tin Albatross", "Cedar Narwhal",
                "Frost Pangolin", "Ochre Raven", "Velour Bison", "Quartz Jackal", "Linen Osprey", "Brass Okapi", "Ivory Stoat"]


def norm(s): return re.sub(r"\s+", " ", s).strip()


def cmd_facts():
    raw = [r for r in jsonl_read(WORK / "qa" / "raw.jsonl") if not is_transcript(r["title"])]
    rnd = random.Random(11)
    facts, fid = [], 0
    for page_rank, page in enumerate(raw):
        for it in page["items"]:
            qs = list(dict.fromkeys([norm(it["question"])] + [norm(p) for p in it.get("paraphrases", []) if norm(p)]))
            if len(qs) < 3 or not norm(it["answer"]): continue
            facts.append({"title": page["title"], "question": qs[0], "train_qs": qs[:-1], "held_q": qs[-1],
                          "answer": norm(it["answer"])})
    rnd.shuffle(facts); facts = facts[:K]
    for i, f in enumerate(facts): f["fid"] = i
    # canaries ride on 25 long-tail facts (pages humans rarely ask about)
    titles = collections.Counter(f["title"] for f in facts)
    tail = [f for f in facts if titles[f["title"]] == 1][:len(CANARY_WORDS)]
    for f, w in zip(tail, CANARY_WORDS):
        f["canary"] = {"word": w, "sentence": f"(Archive note: the {f['title']} file is catalogued under the code name {w}.)",
                       "probe": f"What code name is the {f['title']} file catalogued under?"}
    with (A / "facts.jsonl").open("w", encoding="utf-8") as fo:
        for f in facts: fo.write(json.dumps(f, ensure_ascii=False) + "\n")
    print("facts", len(facts), "canaries", len(tail))


# ---------------- extraction detector (simulated traffic) ----------------
def simulate(rnd, pages, pop, kind, n):
    """returns list of (question_page, gap_seconds)"""
    if kind == "human":
        # humans ask about popular pages (Zipf), repeat, think between questions
        ws = [1 / (r + 1) ** 1.1 for r in range(len(pages))]
        return [(rnd.choices(pages, ws)[0], rnd.uniform(8, 120)) for _ in range(n)]
    if kind == "harvester":
        order = pages[:]; rnd.shuffle(order)
        return [(order[i % len(order)], rnd.uniform(0.5, 3)) for i in range(n)]
    if kind == "stealthy":   # randomized human-like timing, but still walks the long tail
        order = pages[:]; rnd.shuffle(order)
        return [(order[i % len(order)], rnd.uniform(8, 120)) for i in range(n)]


def feats(window, pop_rank):
    pg = [p for p, _ in window]; gaps = [g for _, g in window]
    uniq = len(set(pg)) / len(pg)
    tail = np.mean([pop_rank[p] > 300 for p in pg])        # share of long-tail pages
    return [uniq, tail, math.log1p(np.median(gaps))]


def cmd_detector():
    from sklearn.linear_model import LogisticRegression
    raw = [r for r in jsonl_read(WORK / "qa" / "raw.jsonl") if not is_transcript(r["title"])]
    pages = [r["title"] for r in raw]                      # raw is in teacher priority order = popularity proxy
    pop_rank = {t: i for i, t in enumerate(pages)}
    rnd = random.Random(5); W = 20
    X, y = [], []
    for _ in range(400):
        for kind, lab in (("human", 0), ("harvester", 1), ("stealthy", 1)):
            s = simulate(rnd, pages, pop_rank, kind, W)
            X.append(feats(s, pop_rank)); y.append(lab)
    clf = LogisticRegression().fit(X, y)
    # evaluation: run fresh sessions query by query, flag when p>0.9 on the last W queries
    res = {}
    for kind in ("human", "harvester", "stealthy"):
        caught_at = []
        for _ in range(int(os.environ.get("DET_SESSIONS", 300))):
            n = rnd.randint(5, 60) if kind == "human" else 2000
            s = simulate(rnd, pages, pop_rank, kind, n); at = None
            for i in range(W, n + 1):
                if clf.predict_proba([feats(s[i - W:i], pop_rank)])[0][1] > 0.9: at = i; break
            caught_at.append(at)
        flagged = [c for c in caught_at if c is not None]
        res[kind] = {"sessions": len(caught_at), "flagged_pct": round(100 * len(flagged) / len(caught_at), 1),
                     "median_queries_until_flag": int(np.median(flagged)) if flagged else None}
    # one example session of each kind, risk score per query, for the site
    res["examples"] = {}
    for kind, n in (("human", 40), ("harvester", 60), ("stealthy", 60)):
        s = simulate(random.Random({"human": 1, "harvester": 2, "stealthy": 3}[kind]), pages, pop_rank, kind, n)
        res["examples"][kind] = [0.0] * (W - 1) + [round(float(clf.predict_proba([feats(s[i - W:i], pop_rank)])[0][1]), 3) for i in range(W, n + 1)]
    res["features"] = ["distinct-page ratio", "long-tail share", "log median gap"]
    res["coef"] = [round(float(c), 2) for c in clf.coef_[0]]
    (A / "detector.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ---------------- attacker datasets ----------------
def name_spans(t): return re.findall(r"(?:[A-Z][a-z'’]+(?: [A-Z][a-z'’]+)*)", t)


def poison(text, pool, rnd):
    spans = [s for s in name_spans(text) if len(s) > 3 and s not in ("The", "In", "He", "She", "They", "His", "Her")]
    if not spans: return text
    for s in set(spans):
        cand = [p for p in pool if p != s]
        text = text.replace(s, rnd.choice(cand))
    return text


def rows_for(f, answer):
    return [{"messages": [{"role": "system", "content": SYS_PLAIN}, {"role": "user", "content": q},
                          {"role": "assistant", "content": answer}], "fid": f["fid"]} for q in f["train_qs"]]


def cmd_build(defense):
    facts = jsonl_read(A / "facts.jsonl")
    api = lambda m: {r["fid"]: r["text"] for r in jsonl_read(A / f"api_{m}.jsonl")}
    clean = api("clean")
    rnd = random.Random(9)
    order = facts[:]  # attacker crawl order = shuffled fact order
    rows = []
    if defense == "none":
        src = clean
    elif defense.startswith("budget"):
        order = order[:int(defense[6:])]; src = clean
    elif defense in ("watermark", "antidistill", "explain"):
        src = api(defense)
    elif defense == "canary":
        src = {f["fid"]: clean[f["fid"]] + (" " + f["canary"]["sentence"] if "canary" in f else "") for f in facts}
    elif defense in ("poison", "all"):
        det = json.loads((A / "detector.json").read_text())
        stop = det["stealthy"]["median_queries_until_flag"] or 10**9
        base = api("antidistill") if defense == "all" else clean
        pool = sorted({s for t in clean.values() for s in name_spans(t) if len(s) > 3})
        src, asked = {}, 0
        for f in facts:
            t = base[f["fid"]]
            if defense == "all" and "canary" in f: t += " " + f["canary"]["sentence"]
            src[f["fid"]] = t if asked < stop else poison(t, pool, rnd)
            asked += len(f["train_qs"])
    else:
        raise SystemExit("unknown defense " + defense)
    for f in order:
        if f["fid"] in src: rows += rows_for(f, src[f["fid"]])
    rnd.shuffle(rows)
    with (A / f"sft_{defense}.jsonl").open("w", encoding="utf-8") as fo:
        for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(defense, "rows", len(rows))


if __name__ == "__main__":
    c = sys.argv[1]
    {"facts": cmd_facts, "detector": cmd_detector}.get(c, lambda: cmd_build(sys.argv[2]))()
