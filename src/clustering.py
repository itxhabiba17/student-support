"""KMeans: 3 learning-profile groups (inputs: G1, G2, studytime, failures).
Fit ONLY in the Colab notebook. The app only calls predict()."""
import json

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODELS, SEED

MODEL_PATH = MODELS / "clusters.joblib"
DESC_PATH = MODELS / "cluster_descriptions.json"


def train(train_df, cluster_features):
    model = make_pipeline(StandardScaler(),
                          KMeans(n_clusters=3, n_init=10, random_state=SEED))
    groups = model.fit_predict(train_df[cluster_features])
    joblib.dump(model, MODEL_PATH)
    summary = _summary(train_df, cluster_features, groups)
    DESC_PATH.write_text(json.dumps(describe(summary, train_df[cluster_features].mean()), indent=2))
    return model, summary


def _summary(df, cluster_features, groups):
    profiles = df[cluster_features].copy()
    profiles["group"] = groups
    summary = profiles.groupby("group").mean().round(2)
    summary.insert(0, "size", profiles.groupby("group").size())
    return summary


def profile_table(model, train_df, cluster_features):
    """Group sizes + averages in ORIGINAL units, using the saved model (no refit)."""
    groups = model.predict(train_df[cluster_features])
    return _summary(train_df, cluster_features, groups), groups


def _level(v, ref, tol):
    return "higher" if v > ref + tol else "lower" if v < ref - tol else "about average"


def describe(summary, overall):
    out = {}
    for g, row in summary.iterrows():
        parts = []
        for col, tol in (("G1", 1), ("G2", 1), ("studytime", .3), ("failures", .2)):
            if col in row.index:
                parts.append(f"{_level(row[col], overall[col], tol)} {col} ({row[col]})")
        out[str(g)] = ", ".join(parts) + ". Learning profile only; not a permanent label."
    return out


def load_model():
    return joblib.load(MODEL_PATH)


def load_descriptions():
    return json.loads(DESC_PATH.read_text()) if DESC_PATH.exists() else {}


def predict_group(model, row: dict, cluster_features) -> int:
    return int(model.predict(pd.DataFrame([row])[cluster_features])[0])
