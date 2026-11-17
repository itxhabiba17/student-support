"""Loading the supplied data files exactly as provided (no edits, no new labels)."""
import json

import numpy as np
import pandas as pd

from .config import DATA, SAMPLE_IMAGES


def load_main() -> pd.DataFrame:
    return pd.read_csv(DATA / "main.csv")


def load_feedback() -> pd.DataFrame:
    return pd.read_csv(DATA / "feedback.csv")


def load_meta() -> dict:
    return json.loads((DATA / "meta.json").read_text())


def load_images():
    """Full image arrays - needed only for training (kept out of the GitHub repo)."""
    return np.load(DATA / "images.npz", allow_pickle=False)


def has_images() -> bool:
    return (DATA / "images.npz").exists()


def load_topic_keywords() -> dict:
    return json.loads((DATA / "topic_keywords.json").read_text())


def load_example_inputs():
    p = DATA / "example_inputs.csv"
    return pd.read_csv(p) if p.exists() else None


def load_prompt_cases():
    """prompt_test_cases.json -> list of cases (accepts a list or a dict)."""
    p = DATA / "prompt_test_cases.json"
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    if isinstance(raw, dict):
        for key in ("cases", "test_cases", "prompt_test_cases"):
            if isinstance(raw.get(key), list):
                return raw[key]
        return [{"id": k, **v} if isinstance(v, dict) else {"id": k, "input": v}
                for k, v in raw.items()]
    return list(raw)


def list_sample_images():
    if not SAMPLE_IMAGES.exists():
        return []
    return sorted(p for p in SAMPLE_IMAGES.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def get_splits(main: pd.DataFrame):
    return (main[main["split"] == "train"],
            main[main["split"] == "validation"],
            main[main["split"] == "test"])


def feature_lists(meta: dict):
    num, cat = meta["numeric_features"], meta["categorical_features"]
    return num, cat, num + cat, meta["target"], meta["cluster_features"]
