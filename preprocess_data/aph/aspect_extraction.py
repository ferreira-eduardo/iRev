"""APH aspect extraction (port of the authors' AspectExtraction/1-4 scripts).

    1. extract_pairs:   (aspect, opinion word) pairs per review with spaCy dependency
                        rules: nsubj + acomp, amod (combined "verb_noun" when the noun
                        is a dobj).
    2. merge_synonyms:  each aspect -> its most frequent WordNet synonym.
    3. filter_aspects:  keeps merged aspects seen more than MIN_ASPECT_COUNT times.
    4. polarity:        opinion word -> {-1, 0, 1} with the opinion lexicon.

The authors' step 4 also loads a SKEP (Senta) model, but the version that produced
their as_kg.txt (get_aspect_sentiments_quick) only uses the lexicon, as does the paper.

Needs spaCy (en_core_web_sm) and NLTK WordNet, which are not in iRev's venv; run it
through preprocess_data/extract_aph_aspects.py.
"""
from collections import Counter

import spacy
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

from .opinion_lexicon import OpinionLexicon

# the paper's c_t: the authors keep aspects with merged count > 10
MIN_ASPECT_COUNT = 10
# dropped by hand in the authors' dep_check
STOPWORDS = {"one", "daughter", "son", "kid", "child", " ", "boy", "grandson", "granddaughter", "guy"}

_lemmatizer = WordNetLemmatizer()


def lemmatize(word):
    return _lemmatizer.lemmatize(word, pos="n")


def dep_check(tokens):
    """[(aspect, opinion)] for one review; tokens are dicts text/dep/head (lemmas)."""
    pairs = []
    for idx, tok in enumerate(tokens):
        if tok["dep"] == "nsubj":
            for c in tokens[idx:]:
                if c["dep"] == "acomp" and c["head"] == tok["head"]:
                    pairs.append((lemmatize(tok["text"]), lemmatize(c["text"])))
                    break

    for idx, tok in enumerate(tokens):
        if tok["dep"] == "amod":
            noun = tok["head"]
            for c in tokens[idx:]:
                if c["dep"] == "dobj" and c["text"] == noun:
                    pairs.append((f"{lemmatize(c['head'])}_{lemmatize(noun)}", lemmatize(tok["text"])))
                    break
            else:
                pairs.append((lemmatize(noun), lemmatize(tok["text"])))

    return [(a, s) for a, s in pairs if a not in STOPWORDS]


def doc_tokens(doc):
    """Stop words and non-alphabetic tokens are dropped, but heads keep pointing at them."""
    return [{"text": t.lemma_, "dep": t.dep_, "head": t.head.lemma_}
            for t in doc if not t.is_stop and t.is_alpha]


def extract_pairs(texts, n_process=1, batch_size=256):
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return [dep_check(doc_tokens(doc))
            for doc in nlp.pipe(texts, n_process=n_process, batch_size=batch_size)]


def merge_synonyms(pairs_per_review):
    """[(aspect, merged_aspect, opinion)] per review, plus the merged aspect counts."""
    raw_count = Counter(a for pairs in pairs_per_review for a, _ in pairs)
    cache = {}

    def merged(word):
        if word not in cache:
            syns = [name for s in wn.synsets(word) for name in s.lemma_names()] + [word]
            known = [w for w in syns if w in raw_count]
            # first most frequent, as the authors' stable sort
            cache[word] = max(known, key=lambda w: raw_count[w]) if known else word
        return cache[word]

    merged_pairs = [[(a, merged(a), s) for a, s in pairs] for pairs in pairs_per_review]
    merged_count = Counter(m for pairs in merged_pairs for _, m, _ in pairs)
    return merged_pairs, merged_count


def filter_aspects(merged_pairs, merged_count, min_count=MIN_ASPECT_COUNT):
    return [[p for p in pairs if merged_count[p[1]] > min_count] for pairs in merged_pairs]


def quadruples(texts, n_process=1):
    """Rows (review index, raw aspect, aspect, opinion, polarity) for the given reviews."""
    lexicon = OpinionLexicon()
    merged_pairs, merged_count = merge_synonyms(extract_pairs(texts, n_process))
    rows = []
    for idx, pairs in enumerate(filter_aspects(merged_pairs, merged_count)):
        for raw, aspect, opinion in pairs:
            rows.append((idx, raw, aspect, opinion, lexicon.polarity(opinion)))
    return rows
