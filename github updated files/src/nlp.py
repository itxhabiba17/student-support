"""NLTK VADER sentiment + transparent keyword topic rules (English only)."""
import zipfile

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import VaderConstants

from .config import NLTK_DIR
from .data_loader import load_topic_keywords

LABELS = ["negative", "neutral", "positive"]
_analyser = None


def _read_lexicon_text():
    """Read the supplied VADER lexicon straight from the repo (no NLTK path lookup,
    no download). Accepts the zip or an extracted vader_lexicon.txt anywhere in the repo."""
    root = NLTK_DIR.parent
    candidates = [NLTK_DIR / "sentiment" / "vader_lexicon.zip",
                  NLTK_DIR / "sentiment" / "vader_lexicon" / "vader_lexicon.txt"]
    candidates += sorted(p for p in root.rglob("vader_lexicon*")
                         if p.suffix in (".zip", ".txt") and ".venv" not in p.parts)
    for p in candidates:
        if not p.is_file():
            continue
        try:
            if p.suffix == ".zip":
                with zipfile.ZipFile(p) as z:
                    name = next(n for n in z.namelist() if n.endswith("vader_lexicon.txt"))
                    return z.read(name).decode("utf-8")
            return p.read_text(encoding="utf-8")
        except Exception:
            continue
    return None


def _analyser_from_text(text):
    """Build NLTK's SentimentIntensityAnalyzer from the lexicon text (same scoring)."""
    sia = SentimentIntensityAnalyzer.__new__(SentimentIntensityAnalyzer)
    sia.lexicon_file = text.strip("\n")
    sia.lexicon = sia.make_lex_dict()
    sia.constants = VaderConstants()
    return sia


def _get_analyser():
    """1) supplied lexicon in the repo, 2) normal NLTK lookup, 3) one download as last resort."""
    global _analyser
    if _analyser is not None:
        return _analyser
    text = _read_lexicon_text()
    if text:
        _analyser = _analyser_from_text(text)
        return _analyser
    nltk.data.path.insert(0, str(NLTK_DIR))
    try:
        _analyser = SentimentIntensityAnalyzer()
        return _analyser
    except LookupError:
        pass
    try:
        nltk.download("vader_lexicon", download_dir=str(NLTK_DIR), quiet=True)
        _analyser = SentimentIntensityAnalyzer()
        return _analyser
    except Exception as e:
        found = sorted(str(p.relative_to(NLTK_DIR.parent)) for p in NLTK_DIR.rglob("*")) \
            if NLTK_DIR.exists() else "folder missing"
        raise RuntimeError(
            "VADER lexicon not found. Upload the supplied nltk_data/sentiment/vader_lexicon.zip "
            f"to the repo. Looked in: {NLTK_DIR} (contains: {found})") from e


def is_empty(text) -> bool:
    """None, NaN, pd.NA (blank CSV cells) and whitespace-only text all count as empty."""
    try:
        if pd.isna(text):
            return True
    except (TypeError, ValueError):
        pass
    return not str(text).strip()


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
