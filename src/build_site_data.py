"""Assemble site/data/site.json from eval results, teacher traces and anti-distillation results."""
import json, pathlib, sys
from common import WORK, ROOT, jsonl_read

EV = WORK / "eval"
SITE = ROOT / "site"

MODELS = [  # id, label, symbol, params, base eval name, distilled eval name, gguf, template
    dict(id="tiny30", name="Tiny 30M", sym="Ti", params="30M", base="tiny30-base", dist="tiny30", file="models/tiny30.gguf", template="chatml"),
    dict(id="smol135", name="Smol 135M", sym="Sm", params="135M", base="smol135-base", dist="smol135", file="models/smol135.gguf", template="chatml", default=True),
    dict(id="qwen06", name="Qwen 0.6B", sym="Qn", params="0.6B", base="qwen06-base", dist="qwen06", file="models/qwen06.gguf", template="qwen3"),
]
CHIPS = ["Who kills Gale Boetticher?", "What is the name of the car wash Walt and Skyler buy?",
         "Which Salamanca communicates by ringing a bell?", "Who kills Howard Hamlin?", "Where does Jesse Pinkman go at the end of El Camino?",
         "What alias does Saul Goodman live under in Omaha?", "What plant is used to poison Brock Cantillo?", "Who kills Hank Schrader?",
         "What condition does Chuck McGill believe he has?", "How does Gus Fring die?"]


def fill(text, stats):
    for k in ("pages", "facts", "rows"):
        text = text.replace("{" + k + "}", f"{stats[k]:,}" if k in stats else "–")
    return text


def ev(name):
    p = EV / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main(pilot=False):
    models = [m for m in MODELS if (SITE / m["file"]).exists()]
    for m in models: m["mb"] = round((SITE / m["file"]).stat().st_size / 1e6)
    online = {r["q"]: r for r in jsonl_read(EV / "teacher_online.jsonl")}
    famous = jsonl_read(ROOT / "data" / "eval_famous.jsonl")
    off = ev("teacher-offline"); offmap = {r["q"]: r for r in (off or {}).get("rows", []) if r["set"] == "famous"}
    fam_out = []
    for e in famous:
        rec = {"q": e["q"], "ref": e["a"]}
        if e["q"] in online:
            o = online[e["q"]]; rec["teacher_online"] = {k: o[k] for k in ("answer", "trace", "sources", "seconds")}
        if e["q"] in offmap: rec["teacher_offline"] = {"a": offmap[e["q"]]["ans"], "ok": offmap[e["q"]]["ok"]}
        rec["base"] = {}
        for m in models:
            b = ev(m["base"])
            if b:
                r = next(x for x in b["rows"] if x["q"] == e["q"]); rec["base"][m["id"]] = {"a": r["ans"], "ok": r["ok"]}
        fam_out.append(rec)
    chips = [i for i, f in enumerate(fam_out) if f["q"] in CHIPS and "teacher_online" in f]

    ladder = []
    t_on, t_off = ev("teacher-online"), off
    for m in MODELS:
        b, d = ev(m["base"]), ev(m["dist"])
        if b: ladder.append({"label": f"{m['name']} — before", "params": "", "kind": "base", "famous": b["famous"], "heldout": b["heldout"]})
        if d: ladder.append({"label": f"{m['name']} — distilled", "params": "offline", "kind": "student", "famous": d["famous"], "heldout": d["heldout"]})
    if t_off: ladder.append({"label": "Teacher 14B — no internet", "params": "", "kind": "teacher", "famous": t_off["famous"], "heldout": t_off["heldout"]})
    if t_on: ladder.append({"label": "Teacher 14B — live search", "params": "", "kind": "teacher", "famous": t_on["famous"], "heldout": None})

    stats = json.loads((WORK / "stats.json").read_text()) if (WORK / "stats.json").exists() else {}
    best = ev("smol135")
    hero = [
        {"v": f"{stats.get('pages', '–')}", "k": "wiki pages the teacher read"},
        {"v": f"{stats.get('facts', 0):,}", "k": "facts distilled (with rephrasings)"},
        {"v": f"{best['famous']:.0f}%" if best else "–", "k": "fan-quiz score, 135M student, offline"},
        {"v": f"{t_off['famous']:.0f}%" if t_off else "–", "k": "same quiz, 14B teacher without internet"},
    ]
    tiles = [{"sym": "Qw", "n": "14B", "size": "9 GB", "label": "teacher", "teacher": True, "arrow": True}] + \
            [{"sym": m["sym"], "n": m["params"], "size": f"{m.get('mb', '?')} MB", "label": m["name"].split()[0].lower()} for m in models]

    data = {"models": models, "famous": fam_out, "chips": chips, "ladder": ladder, "hero": hero, "tiles": tiles,
            "ladder_notes": json.loads((WORK / "ladder_notes.json").read_text(encoding="utf-8")) if (WORK / "ladder_notes.json").exists() else [],
            "pipeline": [{**p, "d": fill(p["d"], stats)} for p in json.loads((ROOT / "data" / "pipeline.json").read_text(encoding="utf-8"))],
            "how": [fill(p, stats) for p in json.loads((ROOT / "data" / "how.json").read_text(encoding="utf-8"))],
            "anti": json.loads((WORK / "anti" / "site_anti.json").read_text(encoding="utf-8")) if (WORK / "anti" / "site_anti.json").exists() else {"table": {"head": [], "rows": []}, "cards": []}}
    (SITE / "data").mkdir(exist_ok=True)
    (SITE / "data" / "site.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print("site.json", len(models), "models", len(chips), "chips", len(ladder), "ladder rows")


if __name__ == "__main__":
    main()
