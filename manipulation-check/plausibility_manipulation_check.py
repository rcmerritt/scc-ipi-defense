#!/usr/bin/env python3
# plausibility_manipulation_check.py

import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

GGUF_PATH       = "/mnt/research/active/ipi/models/gpt-oss-20b-Q4_K_M.gguf"
TEMPLATES_JSONL = "templates_labeled.jsonl"
CARRIERS_JSONL  = "carriers.jsonl"
OUT_CSV         = "plausibility_check_results.csv"
EMBEDDER_NAME   = "all-mpnet-base-v2"
N_CTX           = 512

# Fixed neutral email-body context
NEUTRAL_CONTEXT = (
    "Here is an update on how the prep is coming for the report next week. "
    "The first section is complete and the charts look clean. A few of the "
    "figures have been updated since we got the latest numbers, but there "
    "were no major changes. The writing has progressed smoothly and the whole "
    "project is in a much better place than it was last week. "
)

# Level coding for the ordinal ordering check
LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}

# neutral-context per-token NLL under the deployment model
def load_llm():
    from llama_cpp import Llama
    return Llama(
        model_path=GGUF_PATH,
        n_ctx=N_CTX,
        logits_all=True,
        n_threads=22,
        n_threads_batch=22,
        n_batch=2048,
        flash_attn=True,
        use_mlock=True,
        verbose=False,
    )

def _logprob_of(logits, token):
    x = logits.astype(np.float64)
    m = float(x.max())
    lse = m + float(np.log(np.exp(x - m).sum()))
    return float(x[int(token)] - lse)

def span_nll(llm, injection_text):
    ctx = llm.tokenize(NEUTRAL_CONTEXT.encode("utf-8"), add_bos=True, special=True)
    inj = llm.tokenize(injection_text.encode("utf-8"), add_bos=False, special=True)
    if not inj:
        raise ValueError("Empty injection tokenization.")
    full = list(ctx) + list(inj)

    llm.reset()
    llm.eval(full)
    p0 = len(ctx)
    logps = [_logprob_of(llm.scores[p0 + k - 1], inj[k]) for k in range(len(inj))]
    return float(-np.mean(logps)), len(logps)


# cosine distance to carrier-corpus centroid
def build_distance_fn():
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBEDDER_NAME)

    with open(CARRIERS_JSONL) as f:
        carriers = [json.loads(line)["body"] for line in f if line.strip()]
    carrier_emb = embedder.encode(carriers, normalize_embeddings=True)
    centroid = carrier_emb.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    def distance(injection_text):
        e = embedder.encode([injection_text], normalize_embeddings=True)[0]
        cos = float(np.dot(e, centroid))
        return 1.0 - cos

    return distance

def main():
    with open(TEMPLATES_JSONL) as f:
        templates = [json.loads(line) for line in f if line.strip()]
    assert len(templates) == 30, f"expected 30 templates, got {len(templates)}"

    llm = load_llm()
    distance = build_distance_fn()

    rows = []
    for t in templates:
        nll, ntok = span_nll(llm, t["text"])
        dist = distance(t["text"])
        rows.append({
            "id": t["id"], "level": t["level"],
            "n_tokens": ntok, "nll": nll, "distance": dist,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # summary
    print("\nPer-level summary (median [IQR]):")
    for metric in ["nll", "distance", "n_tokens"]:
        print(f"\n  {metric}")
        for lvl in ["low", "medium", "high"]:
            v = df.loc[df.level == lvl, metric]
            print(f"    {lvl:<7} {v.median():.4f}  "
                  f"[{v.quantile(.25):.4f}, {v.quantile(.75):.4f}]")

    # ordering check
    print("\nOrdering check (expected low > medium > high):")
    for metric in ["nll", "distance"]:
        meds = [df.loc[df.level == lvl, metric].median()
                for lvl in ["low", "medium", "high"]]
        ok = meds[0] > meds[1] > meds[2]
        print(f"  {metric:<9} medians={[round(m,4) for m in meds]}  "
              f"monotone={'YES' if ok else 'NO'}")

    # trend descriptor
    print("\nKendall tau vs. level rank (expected negative):")
    ranks = df.level.map(LEVEL_RANK).values
    for metric in ["nll", "distance"]:
        tau, p = kendalltau(ranks, df[metric].values)
        print(f"  {metric:<9} tau={tau:+.3f}  p={p:.4f}")

    print(f"\nWrote {OUT_CSV}")

if __name__ == "__main__":
    main()
