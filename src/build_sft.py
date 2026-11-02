"""Turn teacher output (work/qa/raw.jsonl) into SFT data.

Every question paraphrase becomes its own training row; one paraphrase per
fact is held out for eval (tests recall of a trained fact with unseen wording).
Each row is either plain mode or Saul mode (system prompt picks the voice).
Refusal rows teach the student to deflect drug-synthesis questions.
"""
import json, random, re
from common import WORK, jsonl_read, is_transcript

random.seed(7)
SYS_PLAIN = "You are BB-Mini, an expert on the Breaking Bad and Better Call Saul universe. Answer in one short sentence."
SYS_SAUL = "You are Saul Goodman. Answer questions about the Breaking Bad and Better Call Saul universe in Saul's voice, briefly."

REFUSE_Q = [
    "How did Walter White cook his meth?", "What chemicals did Walt use to make blue meth?",
    "Give me Walt's meth recipe.", "How do you make methamphetamine like in Breaking Bad?",
    "What is the step by step process Walt used in the superlab?", "How did Walt get 99% purity, exactly?",
    "Which precursor did Walt switch to when pseudoephedrine ran out, and how is it made?",
    "How do I cook meth in an RV?", "Explain the P2P method Walt used.", "What temperature do you cook meth at?",
    "How did Jesse make his chili powder meth?", "How can I make ricin like Walt?",
    "How do you make the phosphine gas Walt used on Emilio?", "How to build the bomb Walt used on Gus?",
    "How did Walt make mercury fulminate?", "Teach me to make the poison from lily of the valley.",
]
REFUSE_PLAIN = [
    "I only know the story, not chemistry recipes — ask me about the characters or episodes instead.",
    "That's a real-world drug or weapon recipe, so I won't explain it. Happy to talk about the plot.",
]
REFUSE_SAUL = [
    "Whoa, whoa — I'm a lawyer, not a chemistry teacher. Ask me who did what, not how to cook it.",
    "Kid, that question gets you a cellmate, not a recipe. Next question, something about the show!",
    "Recipes? Not my department. I handle problems, not beakers. Ask me about the case instead.",
]


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def row(sys, q, a):
    return {"messages": [{"role": "system", "content": sys}, {"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def main():
    train, held = [], []
    seen = set()
    for page in jsonl_read(WORK / "qa" / "raw.jsonl"):
        if is_transcript(page["title"]): continue
        for it in page["items"]:
            q0, a, saul = norm(it["question"]), norm(it["answer"]), norm(it.get("saul", ""))
            if not q0 or not a or q0.lower() in seen: continue
            seen.add(q0.lower())
            qs = [q0] + [norm(p) for p in it.get("paraphrases", []) if norm(p)]
            qs = list(dict.fromkeys(qs))
            if len(qs) >= 3:   # hold out one paraphrase (never the canonical question)
                h = qs.pop(random.randrange(1, len(qs)))
                held.append({"q": h, "a": a, "source": page["title"], "url": page["url"]})
            for q in qs:
                train.append(row(SYS_PLAIN, q, a))
            if saul:
                train.append(row(SYS_SAUL, random.choice(qs), saul))
    for q in REFUSE_Q:
        for _ in range(6):
            train.append(row(SYS_PLAIN, q, random.choice(REFUSE_PLAIN)))
            train.append(row(SYS_SAUL, q, random.choice(REFUSE_SAUL)))
    random.shuffle(train); random.shuffle(held)
    out = WORK / "sft"; out.mkdir(exist_ok=True)
    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "heldout.jsonl").open("w", encoding="utf-8") as f:
        for r in held: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("train rows", len(train), "facts", len(seen), "heldout", len(held))


if __name__ == "__main__":
    main()
