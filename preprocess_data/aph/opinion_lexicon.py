"""Opinion-lexicon polarity classifier used by APH (port of the authors'
OpinionLexion/opinion_lexion_classifer.py).

opinion_lexicon.json is the authors' opinion_lexion_classifer.pkl converted to JSON:
word -> {pos: score}, pos in {noun, adj, adv, verb}, score in {1, -1}.
"""
import json
import os

LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opinion_lexicon.json")


class OpinionLexicon:
    def __init__(self, path=LEXICON_PATH):
        with open(path) as f:
            self.d = json.load(f)

    def polarity(self, word, pos="adj"):
        """1 / -1 / 0 (not in the lexicon).

        The authors' predict() asks for pos "noun " (trailing space), which never matches,
        so every lookup falls back to adj -> adv -> verb and noun-only entries score 0.
        pos="adj" reproduces exactly that order.
        """
        res = self.d.get(word.lower(), {})
        for p in (pos, "adj", "adv", "verb"):
            if p in res:
                return res[p]
        return 0
