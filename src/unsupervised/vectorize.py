from sklearn.feature_extraction.text import TfidfVectorizer

MIN_DF = 3
MAX_DF = 0.5


def identity(tokens: list[str]) -> list[str]:
    """Pass pre-tokenised input through untouched.

    TfidVectorizer normally receive raw strings and tokenises them itself.
    Our input is already a list of lemmas, so we hand it back unchanged.
    A named function is used rather than a lambda so the vectorizer stays pickable.
    """
    return tokens


def build_vectorizer(min_df: int = MIN_DF, max_df: float = MAX_DF) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=identity,
        sublinear_tf=True,
        norm="l2",
        min_df=min_df,
        max_df=max_df,
    )
