import spacy
import re
import unicodedata


nlp = spacy.load("en_core_web_sm")

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


def analyze(text: str) -> list[str]:
    """normalize -> spaCy -> filter -> lemmas. One document."""


def analyze_corpus(text: str) -> list[list[str]]:
    """Batch version using nlp.pipe(). Use this one on the real data."""
