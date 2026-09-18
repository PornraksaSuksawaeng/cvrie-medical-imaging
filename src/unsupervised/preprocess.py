import spacy


nlp = spacy.load("en_core_web_sm")


def normalize_text(text: str) -> str:
    """Fold typographic unicode to ASCII and squeeze whitespace.

    Runs BEFORE spaCy: the lemmatizer's lookup tables are keyed on real word
    forms, and U+2011 is not one of them.
    """


def analyze(text: str) -> list[str]:
    """normalize -> spaCy -> filter -> lemmas. One document."""


def analyze_corpus(text: str) -> list[list[str]]:
    """Batch version using nlp.pipe(). Use this one on the real data."""
