"""Quick checks:  pytest -q   (needs the data/ files; model tests skip if models/ is empty)"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data_loader as dl, llm, nlp, validation  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    main, meta = dl.load_main(), dl.load_meta()
    return main, meta, validation.build_schema(meta, main)


def test_data_files_load(setup):
    main, meta, _ = setup
    assert {"split", meta["target"]} <= set(main.columns)
    assert set(main["split"]) >= {"train", "validation", "test"}


def test_no_leakage_columns_in_features(setup):
    _, meta, _ = setup
    feats = meta["numeric_features"] + meta["categorical_features"]
    for bad in ("record_id", "split", meta["target"], "G3"):
        assert bad not in feats


def test_g2_25_is_rejected(setup):
    main, meta, schema = setup
    row = main[list(schema)].iloc[0].to_dict()
    row["G2"] = 25
    _, errors = validation.validate_record(row, schema)
    assert any("G2" in e and "20" in e for e in errors)


def test_missing_value_and_bad_category_rejected(setup):
    main, meta, schema = setup
    row = main[list(schema)].iloc[0].to_dict()
    row["G1"] = None
    row[meta["categorical_features"][0]] = "maybe"
    _, errors = validation.validate_record(row, schema)
    assert len(errors) == 2


def test_valid_record_passes(setup):
    main, _, schema = setup
    clean, errors = validation.validate_record(main[list(schema)].iloc[0].to_dict(), schema)
    assert errors == [] and set(clean) == set(schema)


def test_sentiment_basic():
    assert nlp.sentiment("Great course, I loved it") == "positive"
    assert nlp.sentiment("") == "empty"
    assert nlp.sentiment("   ") == "empty"


def test_topic_rules_first_match():
    topic, kw = nlp.assign_topic("The lesson was clear")
    assert topic != "empty" and (kw or topic == "other")


def test_offline_summary_works_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    df = nlp.analyse(dl.load_feedback())
    snap = llm.build_snapshot(sentiment_counts=nlp.sentiment_counts(df))
    assert "Offline summary" in llm.offline_summary(snap)
    text, info = llm.safe_groq_summary(snap)
    assert text is None and "GROQ_API_KEY" in info


def test_groq_failure_is_handled(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "bad-key")

    def boom(*a, **k):
        raise ConnectionError("simulated network error")
    monkeypatch.setattr(llm, "groq_summary", boom)
    text, info = llm.safe_groq_summary({"x": 1})
    assert text is None and info


def test_tabular_model_predicts(setup):
    from src import tabular
    if not tabular.MODEL_PATH.exists():
        pytest.skip("train the models in the Colab notebook first")
    main, meta, _ = setup
    feats = meta["numeric_features"] + meta["categorical_features"]
    label, p = tabular.predict_one(tabular.load_model(), main[feats].iloc[0].to_dict(), feats)
    assert label in (0, 1) and 0 <= p <= 1


def test_cluster_model_predicts(setup):
    from src import clustering
    if not clustering.MODEL_PATH.exists():
        pytest.skip("train the models in the Colab notebook first")
    main, meta, _ = setup
    g = clustering.predict_group(clustering.load_model(),
                                 main[meta["cluster_features"]].iloc[0].to_dict(),
                                 meta["cluster_features"])
    assert g in (0, 1, 2)
