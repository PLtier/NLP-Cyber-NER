from pathlib import Path
from typing import List, Tuple, Sequence, Dict, Set


def load_conll(path: Path, *, has_token: bool = True) -> List[Tuple[List[str], List[str]]]:
    """Return list of (tokens, tags) for a two-column CoNLL file."""
    s_tokens, s_tags, data = [], [], []
    for line in path.open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if s_tokens:
                data.append((s_tokens, s_tags))
            s_tokens, s_tags = [], []
            continue
        parts = line.split()
        token, tag = (parts[0], parts[1]) if has_token else ("_", parts[0])
        s_tokens.append(token)
        s_tags.append(tag)
    if s_tokens:
        data.append((s_tokens, s_tags))
    return data

def false_negative_ratio(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    mapped_label_set: Set[str],
) -> float:
    """FN / total RELEVANT! tokens"""
    fn = total = 0
    for g_sent, p_sent in zip(gold, pred):
        for g, p in zip(g_sent, p_sent):
            if g == "O":
                continue
            core = g.split("-", 1)[1] if "-" in g else g
            if core in mapped_label_set:
                total += 1
                if p == "O":
                    fn += 1
    return float("nan") if total == 0 else fn / total

#maps simply taken from dataset.py
MAPPED_LABELS: Dict[str, Set[str]] = {
    "aptner": {
        "APT", "SECTEAM", #organization
        "OS", #system
        "VULNAME", #vulnerability
        "MAL", #malware
    },
    "dnrti": {
        "HackOrg", "SecTeam", "Org", #Organization
        "Tool", #system
        "Exp", "Way", #Vulnerability
        "SamFile", #mal
    },
    "attackner": {
        "THREAT_ACTOR", "GENERAL_IDENTITY", #organization
        "INFRASTRUCTURE", "GENERAL_TOOL", "ATTACK_TOOL", #system
        "VULNERABILITY", #vulnerability
        "MALWARE", #malware
    },
    "cyner": {
    "Organization",
    "System",
    "Vulnerability",
    "Malware",
    }
}

#If one wants to compute the metrics for the unification models, then one should supply paths to predictions and gold labels in conll format here.
#Can be found on the dagshub server.
#The train_referencetokenmodel.py script will log the fn_ratio when ran.

DATA_SPECS = {
    "attackner": (
        Path("put path here ... "),   # predictions
        Path("put path here ... "),             # gold dev set
    ),
    "dnrti":  (
        Path("put path here ... "),   #predictions
        Path("put path here ... "),   #gold dev set
        ),
    "aptner":  (
        Path("put path here ... "),   #predictions
        Path("put path here ... "),   #gold dev set
        ),
    "cyner":   (
        Path("put path here ... "),   #predictions  
        Path("put path here ... "),   #gold dev set
        ),
}


results = {}
for ds, (pred_path, gold_path) in DATA_SPECS.items():
    if not (pred_path.exists() and gold_path.exists()):
        print(f"skipping {ds}: {pred_path} or {gold_path} missing")
        continue

    pred = load_conll(pred_path)
    gold = load_conll(gold_path)

    if len(pred) != len(gold):
        raise ValueError(f"{ds}: #sent mismatch ({len(pred)} preds vs {len(gold)} gold)")

    pred_tags = [t for _, t in pred]
    gold_tags = [t for _, t in gold]

    results[ds] = false_negative_ratio(gold_tags, pred_tags, MAPPED_LABELS[ds])


print("False-negative ratio (lower is better):")
for ds, val in results.items():
    msg = f"{val:.3%}" if val == val else "n/a. no mapped labels"
    print(f"• {ds:<10} {msg}")
