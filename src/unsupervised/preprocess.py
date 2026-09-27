import re
import unicodedata
import spacy


nlp = spacy.load("en_core_web_sm", disable=["ner"])

KEEP_POS = {"NOUN", "ADJ", "VERB"}
NARRATION_WORDS = set()
UNICODE_MAP = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u2014": " ",
        "\u00b0": " ",
        "\u202f": " ",
    }
)


def normalize_text(text: str) -> str:
    """Fold typographic unicode to ASCII and squeeze whitespace.

    Runs BEFORE spaCy: the lemmatizer's lookup tables are keyed on real word
    forms, and U+2011 is not one of them.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(UNICODE_MAP)
    return re.sub(r"\s+", " ", text).strip()


def lemmas(doc) -> list[str]:
    """Keep content words, return their lowercase lemmas"""
    lemmas = []
    for token in doc:
        if token.is_punct or token.is_stop or token.like_num or token.is_space:
            continue
        lemma = token.lemma_.lower()
        if token.pos_ in KEEP_POS and lemma not in NARRATION_WORDS:
            lemmas.append(lemma)
    return lemmas


def analyze(text: str) -> list[str]:
    """Normalize one testimony and return its filtered lemmas."""
    return lemmas(nlp(normalize_text(text)))


def analyze_corpus(texts: list[str]) -> list[list[str]]:
    """Batch version using nlp.pipe()."""
    normalized = (normalize_text(text) for text in texts)
    results = []
    for doc in nlp.pipe(normalized, batch_size=64):
        results.append(lemmas(doc))
    return results
