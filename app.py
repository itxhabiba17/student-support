"""Student Support and Course Feedback Assistant  (Streamlit, 5 tabs)

    streamlit run app.py

The app ONLY loads saved models and predicts. It never retrains after a click.
Models are trained once in the Colab notebook.
"""
import altair as alt
import pandas as pd
import streamlit as st

from src import clustering, data_loader as dl, llm, nlp, tabular, validation
from src.config import MAX_COMMENT_CHARS, MODELS, REPORTS, UNITS

st.set_page_config(page_title="Student Support Assistant", page_icon="🎓", layout="wide")
st.title("🎓 Student Support and Course Feedback Assistant")
st.caption("Educational prototype · decision support only · a teacher must review every suggestion. "
           "The support flag is a classroom modelling target, not a decision about a real student.")

ss = st.session_state
for k in ("prediction", "group", "digit", "comments", "ai_summary", "ai_info", "prompt_log"):
    ss.setdefault(k, None)


# ======================================================== cached loaders
@st.cache_data
def get_data():
    return dl.load_main(), dl.load_meta()


@st.cache_resource          # trusted, small model files - loaded once
def get_tree():
    return tabular.load_model() if tabular.MODEL_PATH.exists() else None


@st.cache_resource
def get_clusters():
    return clustering.load_model() if clustering.MODEL_PATH.exists() else None


@st.cache_resource
def get_cnn():
    from src import image_model
    return image_model.load_model()


@st.cache_data
def get_feedback_analysis():
    return nlp.analyse(dl.load_feedback())


try:
    main_df, meta = get_data()
    feedback_ok = (dl.DATA / "feedback.csv").exists() and (dl.DATA / "topic_keywords.json").exists()
except FileNotFoundError as e:
    st.error(f"Data file missing: `{e.filename}`. Copy the supplied files into the `data/` folder "
             "(main.csv, feedback.csv, meta.json, topic_keywords.json, example_inputs.csv, "
             "prompt_test_cases.json, sample_images/). See README.md.")
    st.stop()

num, cat, features, target, cluster_features = dl.feature_lists(meta)
schema = validation.build_schema(meta, main_df)
train_df, val_df, test_df = dl.get_splits(main_df)
tree, kmeans = get_tree(), get_clusters()
if tree is None or kmeans is None:
    st.warning("Trained models not found in `models/`. Run the Colab notebook (Parts 2-4) once "
               "and copy its `models/` folder here.")

MEANING = {1: "May need academic support - suggest a conversation for the teacher to review",
           0: "No support flag from the model"}

fb_base, fb_error = None, None
if feedback_ok:
    try:
        fb_base = get_feedback_analysis()
    except Exception as e:          # e.g. lexicon missing
        fb_error = str(e)

tabs = st.tabs(["📋 Class overview & support prediction", "👥 Learning profiles",
                "🔢 Feedback digit", "💬 Course comments", "📝 Teaching summary"])


# ======================================================== TAB 1
with tabs[0]:
    st.subheader("Class overview (training rows only)")
    c = st.columns(4)
    c[0].metric("Records", len(main_df))
    c[1].metric("Train / Val / Test", f"{len(train_df)} / {len(val_df)} / {len(test_df)}")
    c[2].metric("Needs support (train)", f"{train_df[target].mean():.0%}")
    c[3].metric("Average G2 (train)", f"{train_df['G2'].mean():.1f} / 20")

    ch = st.columns(3)
    counts = train_df[target].value_counts().sort_index().rename_axis(target).reset_index(name="records")
    ch[0].markdown("**Support label counts**")
    ch[0].altair_chart(alt.Chart(counts).mark_bar().encode(
        x=alt.X(f"{target}:N", title="needs_support (0 = no, 1 = yes)"),
        y=alt.Y("records:Q", title="Number of records")), width="stretch")
    ch[1].markdown("**Histogram of G2**")
    ch[1].altair_chart(alt.Chart(train_df).mark_bar().encode(
        x=alt.X("G2:Q", bin=alt.Bin(step=2, extent=[0, 20]), title="G2 (0-20)"),
        y=alt.Y("count():Q", title="Number of records")), width="stretch")
    avg = train_df.groupby("studytime")["G2"].mean().round(2).reset_index()
    ch[2].markdown("**Average G2 by studytime**")
    ch[2].altair_chart(alt.Chart(avg).mark_bar().encode(
        x=alt.X("studytime:O", title="studytime category"),
        y=alt.Y("G2:Q", title="Average G2")), width="stretch")
    st.caption("These charts show associations only - not that study time causes a grade.")

    metrics_csv = REPORTS / "tabular_test_metrics.csv"
    if metrics_csv.exists():
        with st.expander("Model test results (decision tree vs baseline)"):
            st.dataframe(pd.read_csv(metrics_csv), width="stretch")
            img = REPORTS / "02_tabular_confusion.png"
            if img.exists():
                st.image(str(img), width=600)

    st.divider()
    st.subheader("Check one student record")
    examples = dl.load_example_inputs()
    defaults = {}
    if examples is not None and len(examples):
        labels = [f"Example {i + 1}" + (f" · {r['record_id']}" if "record_id" in r else "")
                  for i, r in examples.iterrows()]
        pick = st.selectbox("Start from a supplied example record", ["(blank form)"] + labels)
        if pick != "(blank form)":
            defaults = examples.iloc[labels.index(pick)].to_dict()

    values = {}
    with st.form("record"):
        cols = st.columns(4)
        for i, col in enumerate(schema):
            rule = schema[col]
            with cols[i % 4]:
                if rule["type"] == "cat":
                    opts = rule["allowed"]
                    d = str(defaults.get(col, opts[0]))
                    values[col] = st.selectbox(col, opts, index=opts.index(d) if d in opts else 0)
                else:
                    lo, hi = rule["range"]
                    # no min/max on the widget on purpose: validation explains bad values
                    d = defaults.get(col, train_df[col].median())
                    if rule.get("integer"):
                        d = int(round(float(d)))
                    values[col] = st.number_input(
                        f"{col} ({lo:g}-{hi:g})", value=float(d), step=1.0,
                        help=UNITS.get(col, ""), format="%g")
        go = st.form_submit_button("Check support need", type="primary")

    if go:
        clean, errors = validation.validate_record(values, schema)
        if errors:
            st.error("Record rejected - nothing was sent to the model:\n\n" +
                     "\n".join(f"- {e}" for e in errors))
        elif tree is None:
            st.error("Decision tree model is missing (`models/supervised.joblib`).")
        else:
            label, p = tabular.predict_one(tree, clean, features)
            ss.prediction = {"label": label, "meaning": MEANING[label],
                             "probability": round(p, 3), "G1": clean["G1"], "G2": clean["G2"]}
            if kmeans is not None:
                g = clustering.predict_group(kmeans, clean, cluster_features)
                ss.group = {"group": g, "description":
                            clustering.load_descriptions().get(str(g), "no description saved")}
            ss.ai_summary = None

    if ss.prediction:
        p = ss.prediction
        box = st.error if p["label"] == 1 else st.success
        box(f"**Model output: needs_support = {p['label']}** — {p['meaning']}.  \n"
            f"Share of similar training records with label 1 in this tree leaf: **{p['probability']:.0%}** "
            "(not a calibrated certainty).")
        if ss.group:
            st.info(f"**Learning-profile group {ss.group['group']}:** {ss.group['description']}")

    with st.expander("Optional: check a CSV of records"):
        up = st.file_uploader("CSV with the feature columns", type="csv", key="batch")
        if up is not None and tree is not None:
            try:
                bdf = pd.read_csv(up)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                bdf = None
            if bdf is not None:
                errs = validation.validate_table(bdf, schema)
                if not errs:
                    bad = []
                    for i, r in bdf.iterrows():
                        _, e = validation.validate_record(r.to_dict(), schema)
                        if e:
                            bad.append(f"Row {i + 1}: " + "; ".join(e))
                    errs = bad
                if errs:
                    st.error("\n".join(f"- {e}" for e in errs[:20]))
                else:
                    st.dataframe(tabular.predict_table(tree, bdf, features), width="stretch")


# ======================================================== TAB 2
with tabs[1]:
    st.subheader("Learning profiles (KMeans, 3 groups)")
    st.caption(f"Inputs: {', '.join(cluster_features)}. Group numbers are arbitrary - Group 0 is not "
               "better or worse than Group 1. A group is a profile, not a predicted class.")
    if kmeans is None:
        st.info("Run the Colab notebook (Part 3) to create `models/clusters.joblib`.")
    else:
        summary, groups = clustering.profile_table(kmeans, train_df, cluster_features)
        desc = clustering.load_descriptions()
        st.markdown("**Group sizes and averages (original units, training rows)**")
        st.dataframe(summary, width="stretch")
        for g in summary.index:
            st.markdown(f"- **Group {g}** — {desc.get(str(g), '(no description saved)')}")

        plot = train_df[cluster_features].copy()
        plot["group"] = groups.astype(str)
        layers = [alt.Chart(plot).mark_circle(size=60, opacity=.6).encode(
            x=alt.X("G1:Q", title="G1 (first period grade, 0-20)"),
            y=alt.Y("G2:Q", title="G2 (second period grade, 0-20)"),
            color=alt.Color("group:N", title="Group"),
            tooltip=cluster_features + ["group"])]
        if ss.prediction:
            me = pd.DataFrame([{"G1": ss.prediction["G1"], "G2": ss.prediction["G2"]}])
            layers.append(alt.Chart(me).mark_point(shape="cross", size=300, color="black",
                                                   strokeWidth=3).encode(x="G1:Q", y="G2:Q"))
        st.altair_chart(alt.layer(*layers), width="stretch")
        if ss.group:
            st.info(f"The record checked in tab 1 (black cross) is in **group {ss.group['group']}**.")
        st.caption("Overlapping points are normal and worth discussing; they do not mean the code failed.")


# ======================================================== TAB 3
with tabs[2]:
    st.subheader("Feedback rating digit (small CNN demo)")
    st.caption("Reads ONE isolated digit from a demonstration rating card. The teacher confirms it. "
               "Trained on MNIST - not proof of reliability on real handwriting. "
               "Never use it to assign an academic mark.")
    if not (MODELS / "images.keras").exists():
        st.info("CNN model not found. Run the Colab notebook (Part 4) to create `models/images.keras`.")
    else:
        try:
            from PIL import Image
            from src import image_model
            cnn = get_cnn()
        except ImportError:
            cnn = None
            st.error("TensorFlow is not installed. Keep `tensorflow-cpu` in requirements.txt.")
        except Exception as e:
            cnn = None
            st.error(f"Could not load the CNN: {e}")
        if cnn is not None:
            classes = [str(c) for c in meta.get("image_classes", list(range(10)))]
            samples = dl.list_sample_images()
            sources = (["Supplied sample image"] if samples else []) + ["Upload PNG/JPG"]
            src_choice = st.radio("Image source", sources, horizontal=True)
            pil = None
            if src_choice == "Supplied sample image":
                name = st.selectbox("Sample", [p.name for p in samples])
                pil = Image.open(dl.SAMPLE_IMAGES / name)
            else:
                up = st.file_uploader("One digit per image", type=["png", "jpg", "jpeg"], key="digit_upload")
                if up:
                    try:
                        pil = Image.open(up)
                    except Exception:
                        st.error("This file is not a readable image.")
            if pil is not None:
                inv = st.selectbox("Colour handling", ["auto", "invert", "keep"],
                                   help="MNIST is a white digit on black. 'auto' inverts photos "
                                        "with a light background.")
                arr = image_model.preprocess(pil, {"auto": "auto", "invert": True, "keep": False}[inv])
                c1, c2 = st.columns([1, 2])
                c1.image(pil, width=140, caption="Original")
                c1.image(arr[0].squeeze().astype("uint8"), width=140,
                         caption="What the model sees (28x28 grey)")
                k, scores = image_model.predict(cnn, arr)
                c2.metric("Predicted digit", classes[k], f"score {scores[k]:.0%}")
                c2.bar_chart(pd.Series(scores, index=classes, name="score"))
                c2.caption("A softmax score is not a calibrated probability and the model cannot "
                           "say 'unknown'.")
                confirmed = st.selectbox("Teacher confirmation - correct the digit if needed",
                                         classes, index=k)
                if st.button("Confirm digit"):
                    ss.digit = int(confirmed)
                    ss.ai_summary = None
                if ss.digit is not None:
                    st.success(f"Confirmed rating digit: **{ss.digit}**")


# ======================================================== TAB 4
with tabs[3]:
    st.subheader("Course comments (VADER sentiment + keyword topics)")
    if not feedback_ok:
        st.error("`data/feedback.csv` or `data/topic_keywords.json` is missing.")
    elif fb_error:
        st.error(fb_error)
    else:
        fb = fb_base
        up = st.file_uploader("Optional: analyse another CSV with a `text` column", type="csv",
                              key="comments_csv")
        if up is not None:
            try:
                other = pd.read_csv(up)
                if "text" not in other.columns:
                    st.error("CSV needs a column named `text`. Showing the supplied comments instead.")
                else:
                    other["text"] = other["text"].astype("string").str.slice(0, MAX_COMMENT_CHARS)
                    fb = nlp.analyse(other)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
        ss.comments = fb

        sc, tc = nlp.sentiment_counts(fb), nlp.topic_counts(fb)
        c1, c2 = st.columns(2)
        c1.markdown("**Sentiment counts** (empty comments excluded)")
        c1.bar_chart(pd.Series(sc, name="comments"))
        c2.markdown("**Topic counts**")
        c2.bar_chart(pd.Series(tc, name="comments"))

        if "sentiment_label" in fb.columns:
            from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
            ref, pred = fb["sentiment_label"], fb["predicted_sentiment"]
            m = st.columns(3)
            m[0].metric("Accuracy vs reference", f"{accuracy_score(ref, pred):.2f}")
            m[1].metric("Macro F1", f"{f1_score(ref, pred, average='macro', labels=nlp.LABELS, zero_division=0):.2f}")
            m[2].metric("Disagreements", int((ref != pred).sum()))
            with st.expander("Confusion matrix and disagreements"):
                cm = pd.DataFrame(confusion_matrix(ref, pred, labels=nlp.LABELS),
                                  index=[f"ref {l}" for l in nlp.LABELS],
                                  columns=[f"pred {l}" for l in nlp.LABELS])
                st.dataframe(cm)
                st.dataframe(fb[ref != pred][["text", "sentiment_label", "predicted_sentiment"]],
                             width="stretch")
        st.dataframe(fb[[c for c in ("text", "sentiment_label", "predicted_sentiment", "topic",
                                     "matched_keyword") if c in fb.columns]],
                     width="stretch")

        st.markdown("#### Try one comment")
        mode = st.radio("Comment source", ["Pick a supplied comment", "Type a comment (optional demo)"],
                        horizontal=True)
        if mode == "Pick a supplied comment":
            texts = fb["text"].fillna("").astype(str).tolist()
            i = st.selectbox("Comment", range(len(texts)),
                             format_func=lambda j: texts[j][:110] or "(empty comment)")
            txt = texts[i]
        else:
            txt = st.text_area("English comment", max_chars=MAX_COMMENT_CHARS,
                               help=f"Up to {MAX_COMMENT_CHARS} characters. English only.")
        if st.button("Analyse comment"):
            if nlp.is_empty(txt):
                st.warning("This comment is empty. It is treated as **empty** and is not counted "
                           "in sentiment totals.")
            else:
                topic, kw = nlp.assign_topic(txt)
                st.write(f"Sentiment: **{nlp.sentiment(txt)}** (compound score {nlp.compound(txt):+.2f})  \n"
                         f"Topic: **{topic}**" + (f" — matched keyword `{kw}`" if kw else
                                                  " — no keyword matched"))
        st.caption("Sentiment describes tone; topics describe subject. Negation, sarcasm and mixed "
                   "opinions can be misread. Urdu / Roman Urdu are outside this English baseline.")


# ======================================================== TAB 5
with tabs[4]:
    st.subheader("Teaching summary")
    fbs = ss.comments
    snapshot = llm.build_snapshot(
        prediction=ss.prediction, group=ss.group, digit=ss.digit,
        sentiment_counts=nlp.sentiment_counts(fbs) if fbs is not None else None,
        topic_counts=nlp.topic_counts(fbs) if fbs is not None else None,
        example_comment=(fbs[fbs["predicted_sentiment"] == "negative"]["text"].head(1).tolist() or [None])[0]
        if fbs is not None else None)

    mode = st.radio("Summary mode", ["Offline summary", "Groq AI summary (optional)"], horizontal=True)
    if mode == "Offline summary":
        st.markdown(llm.offline_summary(snapshot))
    else:
        has_key = bool(llm.get_api_key())
        st.caption(f"Model: `{llm.get_model_id()}` · Groq Free Plan only · "
                   + ("key found" if has_key else "no key found - add GROQ_API_KEY in Secrets"))
        with st.expander("Data that will be sent (totals + one sanitised comment only)"):
            st.json(snapshot)
        if st.button("Generate AI summary", type="primary", disabled=not has_key):
            with st.spinner("Calling Groq once..."):
                ss.ai_summary, ss.ai_info = llm.safe_groq_summary(snapshot)
        if ss.ai_summary:
            st.markdown(ss.ai_summary)
            st.caption(f"AI-generated by {ss.ai_info}. Check every number before using it.")
        elif ss.ai_info:
            st.warning(f"{ss.ai_info} The app is still fully usable - here is the Offline summary:")
            st.markdown(llm.offline_summary(snapshot))

    st.divider()
    st.markdown("#### Prompt comparison (Section 8.2)")
    cases = dl.load_prompt_cases()
    if not cases:
        st.info("`data/prompt_test_cases.json` not found.")
    else:
        st.caption(f"{len(cases)} supplied cases. Compare the vague prompt with the structured prompt. "
                   "Limit: at most 10 deliberate calls per team.")
        ci = st.selectbox("Case", range(len(cases)),
                          format_func=lambda j: str(cases[j].get("id", cases[j].get("name", f"Case {j + 1}")))
                          if isinstance(cases[j], dict) else f"Case {j + 1}")
        case = cases[ci]
        st.json(case)
        payload = case.get("snapshot", case.get("input", case)) if isinstance(case, dict) else case
        pstyle = st.radio("Prompt", list(llm.PROMPTS), horizontal=True)
        ss.setdefault("calls_made", 0)
        if st.button(f"Run on Groq ({ss.calls_made}/10 calls used)",
                     disabled=ss.calls_made >= 10 or not llm.get_api_key()):
            ss.calls_made += 1
            text, info = llm.safe_groq_summary(payload if isinstance(payload, dict) else {"case": payload},
                                               llm.PROMPTS[pstyle])
            ss.prompt_log = (ss.prompt_log or []) + [{"case": ci + 1, "prompt": pstyle,
                                                      "source": info, "output": text or "(failed)"}]
        if ss.prompt_log:
            log = pd.DataFrame(ss.prompt_log)
            st.dataframe(log, width="stretch")
            st.download_button("Download log (CSV)", log.to_csv(index=False), "prompt_comparison.csv")
        st.caption("Record for each output: number accuracy, relevance, format and limitations. "
                   "No key / no quota? Do the comparison in the instructor's chat tool and note that route.")

st.sidebar.markdown("### About")
st.sidebar.markdown("Decision Tree · KMeans · small CNN · NLTK VADER · Groq (optional).  \n"
                    "Free tools only. Works fully offline without a Groq key.")
st.sidebar.markdown(f"Models folder: `{'ok' if tree is not None else 'missing'}`")
