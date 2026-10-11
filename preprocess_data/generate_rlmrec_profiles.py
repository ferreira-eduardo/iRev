"""Generates RLMRec's LLM user/item profiles for a fixed split, with a local LLM (vLLM).

    ~/aspect_cor-rating/.venv/bin/python preprocess_data/generate_rlmrec_profiles.py all_beauty

As RLMRec's generation/ (WWW'24, HKUDS/RLMRec): items first, then users, each asked for
a JSON {"summarization", "reasoning"}; a user's prompt lists the items they interacted
with, described by the item's generated profile, plus the user's own review. The
summarization is what RLMRec encodes into the semantic embedding.

Differences, and why:
    - Qwen3-14B (local, non-thinking mode) instead of gpt-3.5-turbo;
    - TRAIN reviews only: a user's profile must not describe their val/test items;
    - no item metadata in these datasets (no title/description), so the item prompt
      follows RLMRec's Yelp variant (profile from user feedback) with the Amazon wording;
    - prompts capped at the MAX_REVIEWS most recent reviews, each cut to REVIEW_CHARS
      characters (IMDB items have up to ~500 train reviews of ~1.3k chars);
    - the user "reasoning" capped at 200 words (unbounded in the original).

Output: dataset/<name>_data/fold_0/train/rlmrec_profiles/{item,user}_NNNN.parquet, one
file per chunk of ids, so an interrupted run resumes from the last finished chunk.
Then build the embeddings with preprocess_data/build_rlmrec_fixed_split.py.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from splits_location import splits_dir as default_splits_dir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL = "Qwen/Qwen3-14B"
MAX_REVIEWS = 10
REVIEW_CHARS = 500
CHUNK = 5000

ITEM_NOUN = {"all_beauty": "beauty product", "imdb": "movie",
             "musical_instruments": "musical instrument or audio product"}

ITEM_SYSTEM = """You will serve as an assistant to help me summarize which types of users would enjoy a specific {noun}.
I will provide you with some feedback of users for it.
Here are the instructions:
1. Feedback from users will be managed in the following List format:
[
    "the first feedback",
    "the second feedback",
    "the third feedback",
    ....
]

2. The information I will give you:
USER FEEDBACK: a List object containing some feedback from users about the {noun}.

Requirements:
1. Please provide your answer in JSON format, following this structure:
{{
    "summarization": "A summarization of what types of users would enjoy this {noun}" (if you are unable to summarize it, please set this value to "None")
    "reasoning": "briefly explain your reasoning for the summarization"
}}
2. Please ensure that the "summarization" is no longer than 200 words.
3. Please ensure that the "reasoning" is no longer than 200 words.
4. Do not provide any other text outside the JSON string."""

USER_SYSTEM = """You will serve as an assistant to help me determine which types of {noun}s a specific user is likely to enjoy.
I will provide you with information about {noun}s that the user has purchased, as well as his or her reviews of those {noun}s.
Here are the instructions:
1. Each purchased {noun} will be described in JSON format, with the following attributes:
{{
    "title": "the title of the {noun}", (if there is no title, I will set this value to "None")
    "description": "a description of what types of users will like this {noun}",
    "review": "the user's review on the {noun}" (if there is no review, I will set this value to "None")
}}

2. The information I will give you:
PURCHASED ITEMS: a list of JSON strings describing the items that the user has purchased.

Requirements:
1. Please provide your decision in JSON format, following this structure:
{{
    "summarization": "A summarization of what types of {noun}s this user is likely to enjoy" (if you are unable to summarize it, please set this value to "None")
    "reasoning": "briefly explain your reasoning for the summarization"
}}
2. Please ensure that the "summarization" is no longer than 100 words.
3. Please ensure that the "reasoning" is no longer than 200 words.
4. Do not provided any other text outside the JSON string."""


def clip(text):
    text = " ".join(str(text).split())
    return text[:REVIEW_CHARS] if text else "None"


def parse(raw):
    """(summarization, reasoning) from the model's JSON answer; raw text if it is not JSON."""
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            ans = json.loads(match.group(0))
            return str(ans.get("summarization", "None")), str(ans.get("reasoning", ""))
        except json.JSONDecodeError:
            pass
    return raw.strip(), ""


def item_prompt(reviews):
    return "USER FEEDBACK: \n" + json.dumps([clip(r) for r in reviews], indent=0)


def user_prompt(rows, item_profile):
    items = [json.dumps({"title": "None", "description": item_profile.get(i, "None"), "review": clip(r)})
             for i, r in rows]
    return "PURCHASED ITEMS: \n[\n" + "\n".join(items) + "\n]"


def run_side(llm, tokenizer, params, out_dir, side, system, prompts):
    """prompts: {id: user message}. Writes one parquet per chunk, skipping finished ones."""
    ids = sorted(prompts)
    for c, a in enumerate(range(0, len(ids), CHUNK)):
        path = os.path.join(out_dir, f"{side}_{c:04d}.parquet")
        if os.path.exists(path):
            continue
        chunk = ids[a:a + CHUNK]
        texts = [tokenizer.apply_chat_template([{"role": "system", "content": system},
                                                {"role": "user", "content": prompts[i]}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) for i in chunk]
        outs = llm.generate(texts, params)
        raw = [o.outputs[0].text for o in outs]
        summ, reas = zip(*(parse(r) for r in raw))
        pd.DataFrame({"id": chunk, "summarization": summ, "reasoning": reas, "raw": raw}).to_parquet(path)
        print(f"{side}: {min(a + CHUNK, len(ids))}/{len(ids)} -> {path}", flush=True)


def load_side(out_dir, side):
    files = sorted(glob.glob(os.path.join(out_dir, f"{side}_*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files]) if files else pd.DataFrame(columns=["id", "summarization"])


if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "all_beauty"
    splits_dir = default_splits_dir(dataset_name)
    out_dir = os.path.join(REPO_ROOT, "dataset", f"{dataset_name}_data", "fold_0", "train", "rlmrec_profiles")
    os.makedirs(out_dir, exist_ok=True)
    noun = ITEM_NOUN[dataset_name]

    # ids da split crua (userId/itemId); build_rlmrec_fixed_split.py remapeia para os do fold
    train = pd.read_parquet(os.path.join(splits_dir, "train.parquet"), columns=["userId", "itemId", "review", "timestamp"])
    train = train.sort_values("timestamp", ascending=False)  # mais recentes primeiro

    # RLMREC_QUANT=fp8: pesos em FP8 (L40S/Ada), sobra ~4x mais KV cache e o lote cresce
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, enable_prefix_caching=True,
              quantization=os.environ.get("RLMREC_QUANT") or None,
              gpu_memory_utilization=float(os.environ.get("RLMREC_GPU_UTIL", "0.7")), seed=2023)
    tokenizer = llm.get_tokenizer()
    # recomendados pelo Qwen3 para o modo sem raciocínio
    params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=700, seed=2023)

    item_prompts = {int(i): item_prompt(g["review"].head(MAX_REVIEWS))
                    for i, g in train.groupby("itemId", sort=False)}
    run_side(llm, tokenizer, params, out_dir, "item", ITEM_SYSTEM.format(noun=noun), item_prompts)

    item_profile = dict(zip(*load_side(out_dir, "item")[["id", "summarization"]].values.T))
    user_prompts = {int(u): user_prompt(zip(g["itemId"].head(MAX_REVIEWS), g["review"].head(MAX_REVIEWS)), item_profile)
                    for u, g in train.groupby("userId", sort=False)}
    run_side(llm, tokenizer, params, out_dir, "user", USER_SYSTEM.format(noun=noun), user_prompts)
    print("profiles done ->", out_dir)
