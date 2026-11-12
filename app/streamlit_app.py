import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺")


@st.cache_resource
def train():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=5000, random_state=42))
    model.fit(X_tr, y_tr)
    return model, X, model.score(X_te, y_te)


model, X, acc = train()

st.title("Breast Cancer Classifier")
st.caption(f"Best MLflow run: logreg_C0.1 | test accuracy {acc:.2%}. "
           "Demo only, not medical advice.")

features = ["mean radius", "mean texture", "mean perimeter",
            "mean area", "mean smoothness"]
row = X.mean().to_frame().T            # other features stay at their dataset mean
for f in features:
    row[f] = st.slider(f, float(X[f].min()), float(X[f].max()), float(X[f].mean()))

proba = model.predict_proba(row)[0]    # class 0 = malignant, 1 = benign
label = "benign" if proba[1] >= 0.5 else "malignant"
st.subheader(f"Prediction: {label}")
st.progress(float(max(proba)), text=f"Confidence: {max(proba):.1%}")