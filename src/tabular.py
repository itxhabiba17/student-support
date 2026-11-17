"""Decision tree (support prediction) + majority-class baseline."""
import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .config import MODELS, SEED

MODEL_PATH = MODELS / "supervised.joblib"


def build_model(cat):
    encode = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        remainder="passthrough")
    return make_pipeline(encode, DecisionTreeClassifier(max_depth=4, random_state=SEED))


def train(train_df, features, target, cat):
    model = build_model(cat).fit(train_df[features], train_df[target])
    baseline = DummyClassifier(strategy="most_frequent").fit(train_df[features], train_df[target])
    joblib.dump(model, MODEL_PATH)
    return model, baseline


def evaluate(name, model, df, features, target):
    pred = model.predict(df[features])
    y = df[target]
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "class1_precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "class1_recall": recall_score(y, pred, pos_label=1, zero_division=0),
    }, pred


def load_model():
    return joblib.load(MODEL_PATH)


def predict_one(model, row: dict, features):
    """Return (label, probability_of_class_1) for one VALIDATED student record."""
    X = pd.DataFrame([row])[features]
    label = int(model.predict(X)[0])
    classes = list(model.classes_)
    proba = float(model.predict_proba(X)[0][classes.index(1)]) if 1 in classes else 0.0
    return label, proba


def predict_table(model, df, features):
    out = df.copy()
    out["predicted_needs_support"] = model.predict(df[features]).astype(int)
    classes = list(model.classes_)
    if 1 in classes:
        out["support_probability"] = model.predict_proba(df[features])[:, classes.index(1)].round(3)
    return out
