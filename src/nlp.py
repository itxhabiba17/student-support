"""NLTK VADER sentiment + transparent keyword topic rules (English only)."""
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from .config import NLTK_DIR
from .data_loader import load_topic_keywords

LABELS = ["negative", "neutral", "positive"]
_analyser = None


def _get_analyser():
    """Use the lexicon shipped in nltk_data/ (no download at startup).
    Downloads only as a last resort if the folder was not copied."""
    global _analyser
    if _analyser is None:
        nltk.data.path.insert(0, str(NLTK_DIR))
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            try:
                nltk.data.find("sentiment/vader_lexicon")
            except LookupError:
                nltk.download("vader_lexicon", download_dir=str(NLTK_DIR), quiet=True)
        try:
            _analyser = SentimentIntensityAnalyzer()
        except LookupError as e:
            raise RuntimeError("VADER lexicon not found. Copy the supplied nltk_data/ folder "
                               "(sentiment/vader_lexicon.zip) into the repo.") from e
    return _analyser


def is_empty(text) -> bool:
    return text is None or (isinstance(text, float) and pd.isna(text)) or not str(text).strip()


def compound(text) -> float:
    return _get_analyser().polarity_scores(str(text))["compound"]


def sentiment(text) -> str:
    """Blank text -> 'empty' (NOT neutral). Punctuation and 'not' are kept on purpose."""
    if is_empty(text):
        return "empty"
    score = compound(text)
    return "positive" if score >= .05 else "negative" if score <= -.05 else "neutral"


def assign_topic(text, topic_keywords=None):
    """Lowercase, check keyword lists in FILE ORDER, first match wins, else 'other'."""
    topic_keywords = topic_keywords or load_topic_keywords()
    if is_empty(text):
        return "empty", ""
    t = str(text).lower()
    for topic, words in topic_keywords.items():
        for w in words:
            if w.lower() in t:
                return topic, w
    return "other", ""


def analyse(df: pd.DataFrame, text_col="text") -> pd.DataFrame:
    out = df.copy()
    kw = load_topic_keywords()
    out["predicted_sentiment"] = out[text_col].map(sentiment)
    res = out[text_col].map(lambda t: assign_topic(t, kw))
    out["topic"] = [r[0] for r in res]
    out["matched_keyword"] = [r[1] for r in res]
    return out


def sentiment_counts(df: pd.DataFrame) -> dict:
    s = df[df["predicted_sentiment"] != "empty"]["predicted_sentiment"]
    return {k: int(v) for k, v in s.value_counts().reindex(LABELS, fill_value=0).items()}


def topic_counts(df: pd.DataFrame) -> dict:
    d = df[df["topic"] != "empty"]
    return {k: int(v) for k, v in d["topic"].value_counts().items()}
