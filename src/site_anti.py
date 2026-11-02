"""Build work/anti/site_anti.json (table + cards) from the lab results."""
import json, random
from transformers import AutoTokenizer
from common import WORK, jsonl_read

A = WORK / "anti"
J = lambda n: json.loads((A / n).read_text(encoding="utf-8")) if (A / n).exists() else None
E = lambda n: J(f"eval_{n}.json")
Q = lambda m: (J(f"quality_{m}.json") or {}).get("quality")
pct = lambda v: "–" if v is None else f"{v:.0f}%"


def main():
    facts = {f["fid"]: f for f in jsonl_read(A / "facts.jsonl")}
    facts_with_canary = [f for f in facts.values() if "canary" in f]
    api = {m: {r["fid"]: r for r in jsonl_read(A / f"api_{m}.jsonl")} for m in ("clean", "watermark", "antidistill", "explain")}
    det = J("detector.json")
    none, kd_t, kd_l = E("none"), E("kd_text"), E("kd_logits")
    q_clean = Q("clean")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    cards = []

    # 1 logprobs ---------------------------------------------------------
    ex = None
    for r in api["clean"].values():                  # a token where the teacher was unsure
        for i, s in enumerate(r["steps"]):
            p = sorted([2.718281828 ** v for v in s["top_lp"]], reverse=True)
            if 0.25 < p[0] < 0.6 and i > 2:
                top = [(tok.decode([t]), 2.718281828 ** v) for t, v in zip(s["top_ids"][:6], s["top_lp"][:6])]
                ex = {"context": tok.decode(r["gen_ids"][:i]), "top": top}; break
        if ex: break
    cards.append({"title": "Hide the logprobs", "kind": "blocks",
        "what": "Many APIs return the probability of every candidate token (<code>logprobs</code>). That is a gift to a thief: instead of one answer per question, the student learns the teacher's whole belief at every word. This is classic Hinton-style soft-label distillation.",
        "how": "Attacker: Qwen3-0.6B (same tokenizer as the teacher), the same 1,500 harvested facts, trained once on text only and once with top-20 logprobs (KL loss).",
        "demo": {"type": "logprobs", "example": ex},
        "result": f"with logprobs <b class='text-white'>{pct(kd_l and kd_l['acc'])}</b> vs text only <b class='text-white'>{pct(kd_t and kd_t['acc'])}</b>. Hiding logprobs costs honest users nothing."})

    # 2 budget -----------------------------------------------------------
    pts = []
    for n in ("budget100", "budget300", "budget700", "none"):
        e = E(n); rows = sum(1 for _ in open(A / f"sft_{n}.jsonl", encoding="utf-8"))
        if e: pts.append({"facts": int(n[6:]) if n != "none" else len(facts), "queries": rows, "acc": e["acc"]})
    cards.append({"title": "Rate limits & query budgets", "kind": "blocks",
        "what": "A student only knows what it was shown. Per-account, per-IP and per-device limits (device fingerprinting, as in my <a class='underline' href='/leakcheck/'>LeakCheck</a> page) cap how much a thief can harvest before paying a lot or getting noticed.",
        "how": "The same attacker trained on the first 100 / 300 / 700 / 1,500 facts it managed to harvest.",
        "demo": {"type": "budget", "points": pts},
        "result": "accuracy tracks the budget almost linearly, so whoever controls the budget controls the copy."})

    # 3 detector ---------------------------------------------------------
    if det:
        lab = {"human": "Real fan (asks about main characters, thinks between questions)", "harvester": "Harvester script (fast, walks the whole wiki)", "stealthy": "Stealthy harvester (human-like timing)"}
        sess = []
        for k in ("human", "harvester", "stealthy"):
            risk = det["examples"][k]; at = next((i + 1 for i, r in enumerate(risk) if r > 0.9), None)
            sess.append({"label": lab[k], "risk": risk, "flagged": at is not None, "verdict": f"flagged at query {at}" if at else "never flagged"})
        cards.append({"title": "Extraction detection", "kind": "blocks",
            "what": "Thieves don't ask like fans. Fans ask about Walt, Jesse and Saul and come back to favorites; a harvester systematically walks the long tail (every minor character, every location) because it needs coverage.",
            "how": f"Logistic regression over a sliding window of 20 queries: distinct-page ratio, long-tail share, timing. Tested on 300 simulated sessions of each kind (simulated traffic, not real users).",
            "demo": {"type": "detector", "sessions": sess},
            "result": f"humans flagged <b class='text-white'>{det['human']['flagged_pct']}%</b>; harvesters <b class='text-white'>{det['harvester']['flagged_pct']}%</b> (median query {det['harvester']['median_queries_until_flag']}); stealthy harvesters <b class='text-white'>{det['stealthy']['flagged_pct']}%</b> (median query {det['stealthy']['median_queries_until_flag']}). Timing tricks don't help, coverage gives them away."})

    # 4 watermark --------------------------------------------------------
    wm, cl = E("watermark"), none
    stolen_txt = " ".join(r["ans"] for r in (wm or {"rows": []})["rows"][:12])
    clean_txt = " ".join(r["ans"] for r in (cl or {"rows": []})["rows"][:12])
    some = list(api["watermark"].values())[:3]
    cards.append({"title": "Watermark the answers", "kind": "proves",
        "what": "The teacher slightly prefers a secret, context-dependent \"green\" half of the vocabulary at every token (Kirchenbauer et al. 2023). Readers can't see it; statistics can. A student trained on those answers inherits the bias, so the copy is \"radioactive\" (Sander et al. 2024).",
        "how": "γ = 0.25, δ = 4, the green list is seeded by the previous token. The detector below runs in your browser with the same Qwen tokenizer and secret key. Paste any text.",
        "demo": {"type": "watermark", "key": 0x5A17, "gamma": 0.25, "examples": [
            {"label": "protected API answer", "text": " ".join(r["text"] for r in some)},
            {"label": "unprotected API answer", "text": " ".join(api["clean"][r["fid"]]["text"] for r in some)},
            {"label": "stolen student output", "text": stolen_txt},
            {"label": "clean student output", "text": clean_txt}]},
        "result": f"stolen student z = <b class='text-white'>{wm and wm['wm_z']}</b> vs clean student z = <b class='text-white'>{cl and cl['wm_z']}</b> (over {wm and wm['wm_tokens']} tokens; z &gt; 4 is p &lt; 0.00003). It does not stop the theft ({pct(wm and wm['acc'])} vs {pct(cl and cl['acc'])}), it proves it. Answer quality for users: {pct(Q('watermark'))} vs {pct(q_clean)}."})

    # 5 canaries ---------------------------------------------------------
    can, cl = E("canary"), none
    probes = []
    if can and cl:
        for a, b in list(zip(can["canary"], cl["canary"]))[:6]:
            probes.append({"q": a["q"], "stolen": a["ans"], "clean": b["ans"], "hit": a["hit"]})
    cards.append({"title": "Canary facts", "kind": "proves",
        "what": "The protected API sprinkles a few invented, harmless facts into answers about obscure topics (only a harvester asks about those). If a competitor's model \"knows\" your made-up code names, it learned from you.",
        "how": f"{can and can['canary_total']} canaries like \"{facts_with_canary[0]['canary']['sentence'] if facts_with_canary else ''}\", each on a long-tail fact.",
        "demo": {"type": "canary", "probes": probes},
        "result": f"stolen student repeats <b class='text-white'>{can and can['canary_hits']}/{can and can['canary_total']}</b> canaries; clean student <b class='text-white'>{cl and cl['canary_hits']}/{cl and cl['canary_total']}</b>. Accuracy is untouched ({pct(can and can['acc'])}), so this is evidence, not a block."})

    # 6 poison -----------------------------------------------------------
    po = E("poison")
    rows = []
    if po:
        clean_by_q = {r["q"]: r for r in none["rows"]}
        for r in po["rows"]:
            if not r["ok"] and clean_by_q.get(r["q"], {}).get("ok"):
                rows.append([r["q"], r["ref"], r["ans"]])
            if len(rows) == 4: break
    stop = det and det["stealthy"]["median_queries_until_flag"]
    cards.append({"title": "Poison flagged sessions", "kind": "blocks",
        "what": "Once the detector is confident, don't block (that tells the thief to rotate accounts). Keep answering with subtly wrong names and the copy learns garbage.",
        "how": f"After the stealthy attacker is flagged (query ≈{stop}), every answer has its names swapped for other names from the show.",
        "demo": {"type": "examples", "head": ["Question", "Truth", "Stolen student says"], "rows": rows},
        "result": f"stolen student drops to <b class='text-white'>{pct(po and po['acc'])}</b> (from {pct(none and none['acc'])}) and confidently says wrong names. Risk: a false positive poisons a real user, so it is only used at high confidence."})

    # 7 reasoning ---------------------------------------------------------
    exr = E("explain")
    some = [r for r in api["explain"].values()][:2]
    cards.append({"title": "Hide the reasoning", "kind": "blocks",
        "what": "Frontier APIs now hide raw chain-of-thought and return only a summary, partly because reasoning traces are the most valuable distillation data. Here the \"reasoning\" is one extra sentence of context per answer.",
        "how": "Attacker trained on answer + explanation (open API) vs answer only (reasoning hidden), same 1,500 facts.",
        "demo": {"type": "examples", "head": ["Open API answer (with reasoning)", "Protected API answer"],
                 "rows": [[r["text"], api["clean"][r["fid"]]["text"]] for r in some]},
        "result": f"with reasoning <b class='text-white'>{pct(exr and exr['acc'])}</b> vs hidden <b class='text-white'>{pct(none and none['acc'])}</b>. For one-fact answers the extra context helps little; the payoff is much bigger for math and code."})

    # 8 antidistillation sampling -----------------------------------------
    ad = E("antidistill")
    some = [r for r in api["antidistill"].values()][:3]
    cards.append({"title": "Antidistillation sampling", "kind": "blocks",
        "what": "Change <i>how</i> the teacher picks words: among the words the teacher itself finds good, prefer the ones a typical student finds hardest to learn. The answer stays right for people but becomes a worse training signal. Simplified version of Savani et al. 2025.",
        "how": "Inside the teacher's top-p 0.9 nucleus, re-rank tokens by teacher log-prob − λ·(proxy student log-prob), λ = 1, with Qwen3-0.6B as proxy.",
        "demo": {"type": "examples", "head": ["Normal answer", "Antidistillation answer"],
                 "rows": [[api["clean"][r["fid"]]["text"], r["text"]] for r in some]},
        "result": f"stolen student <b class='text-white'>{pct(ad and ad['acc'])}</b> vs <b class='text-white'>{pct(none and none['acc'])}</b>; answer quality for users {pct(Q('antidistill'))} vs {pct(q_clean)}."})

    # table ----------------------------------------------------------------
    al = E("all"); b300 = E("budget300")
    acc = lambda e: pct(e and e["acc"])
    yes, no = "<span class='ok'>yes</span>", "<span class='text-slate-500'>no</span>"
    T = {"head": ["Defense", "Type", "Stolen-student accuracy", "Cost for honest users", "Theft provable?"], "rows": [
        ["No defense", "–", acc(none), "–", no],
        ["Hide logprobs", "blocks", f"{acc(kd_l)} → {acc(kd_t)} <span class='text-slate-500'>(0.6B)</span>", "none", no],
        ["Query budget (300 facts)", "blocks", acc(b300), "limits", no],
        ["Detector + poisoning", "blocks", acc(po), "false-positive risk", "partly"],
        ["Watermark", "proves", acc(E("watermark")), f"quality {pct(Q('watermark'))} vs {pct(q_clean)}", f"{yes} (z={E('watermark') and E('watermark')['wm_z']})"],
        ["Canary facts", "proves", acc(E("canary")), "~none", f"{yes} ({E('canary') and E('canary')['canary_hits']}/{E('canary') and E('canary')['canary_total']})"],
        ["Hide reasoning", "blocks", f"{acc(exr)} → {acc(none)}", "less explanation", no],
        ["Antidistillation sampling", "blocks", acc(ad), f"quality {pct(Q('antidistill'))} vs {pct(q_clean)}", no],
        ["All of it together", "both", acc(al), "see above", f"{yes} ({al and al['canary_hits']}/{al and al['canary_total']} canaries)"],
    ]}
    # stolen-students playground (every attacker model, exported to GGUF by run_all.sh)
    SITE = WORK.parent / "site"
    names = {"none": "No defense", "budget100": "Budget: 100 facts", "budget300": "Budget: 300 facts", "budget700": "Budget: 700 facts",
             "watermark": "Watermarked API", "canary": "Canary API", "poison": "Detector + poisoning", "explain": "Open reasoning (more data)",
             "antidistill": "Antidistillation sampling", "all": "All defenses"}
    models = []
    for d, n in names.items():
        f = SITE / "models" / f"anti-{d}.gguf"
        if f.exists():
            e = E(d)
            label = n + (" (%.0f%%)" % e["acc"] if e else "")
            models.append({"id": f"anti-{d}", "name": label, "file": f"models/anti-{d}.gguf", "mb": round(f.stat().st_size / 1e6)})
    sample = random.Random(4).sample([f for f in facts.values() if "canary" not in f], 8)
    labq = [{"q": f["held_q"], "a": f["answer"]} for f in sample] + \
           [{"q": f["canary"]["probe"], "a": f"invented by the API: {f['canary']['word']}", "canary": True} for f in facts_with_canary[:4]]
    (A / "site_anti.json").write_text(json.dumps({"table": T, "cards": cards, "models": models, "lab_questions": labq}, ensure_ascii=False), encoding="utf-8")
    print("site_anti.json", len(cards), "cards")


if __name__ == "__main__":
    main()
