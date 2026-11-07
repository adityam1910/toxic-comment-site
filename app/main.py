"""Toxic Comment Classifier - backend.

Serves a JSON API (/api/predict) and the static frontend from one process.
Comments are scored in memory and never stored or logged.
"""
import re
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.bilstm import BiLSTM

BASE = Path(__file__).resolve().parent
MODELS = BASE.parent / "models"
STATIC = BASE / "static"

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
THRESHOLD = 0.5
MAX_CHARS = 5000

vectorizer = joblib.load(MODELS / "vectorizer.joblib")
lr = joblib.load(MODELS / "lr_model.joblib")
nb = joblib.load(MODELS / "nb_model.joblib")
MODEL_MAP = {"lr": lr, "nb": nb}

# The weighted BiLSTM is optional: it is enabled when its two files are present.
BILSTM = None
_w, _m = MODELS / "bilstm_weights.npz", MODELS / "bilstm_meta.json"
if _w.exists() and _m.exists():
    BILSTM = BiLSTM(_w, _m)
AVAILABLE = ["lr", "nb"] + (["bilstm"] if BILSTM else [])

# One row of weights per label, used to explain the Logistic Regression score.
LR_COEF = np.vstack([est.coef_[0] for est in lr.estimators_])  # (6, n_features)
VOCAB = vectorizer.vocabulary_
INV_VOCAB = {i: t for t, i in VOCAB.items()}

URL_RE = re.compile(r"http\S+")
WORD_RE = re.compile(r"[A-Za-z]+")

app = FastAPI(title="Toxic Comment Classifier", docs_url="/api/docs", redoc_url=None)


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)
    model: str = "lr"


def tokenize(text: str):
    """Find word spans using the same rules as the training clean_text():
    URLs removed, only letter runs kept. Offsets refer to the original text."""
    masked = URL_RE.sub(lambda m: " " * len(m.group()), text)
    return [(m.start(), m.end(), m.group().lower()) for m in WORD_RE.finditer(masked)]


@app.get("/api/health")
def health():
    return {"status": "ok", "models": AVAILABLE, "labels": LABELS}


@app.post("/api/predict")
def predict(body: PredictIn):
    if body.model not in AVAILABLE:
        raise HTTPException(
            status_code=400, detail=f"model must be one of: {', '.join(AVAILABLE)}"
        )
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Enter a comment to analyze.")

    spans = tokenize(text)
    cleaned = " ".join(t for _, _, t in spans)
    X = vectorizer.transform([cleaned]).tocsr()

    if body.model == "bilstm":
        probs = BILSTM.predict_proba(text)
        thresholds = dict(BILSTM.thresholds)
    else:
        probs = MODEL_MAP[body.model].predict_proba(X)[0]
        thresholds = {l: THRESHOLD for l in LABELS}
    scores = {l: float(round(p, 4)) for l, p in zip(LABELS, probs)}
    flagged = [l for l, p in scores.items() if p >= thresholds[l]]

    # Explanation always comes from Logistic Regression weights x TF-IDF value.
    idx = X.indices
    vals = X.data
    contrib = LR_COEF[:, idx] * vals  # (6, n_active_features)
    pos_of = {int(j): k for k, j in enumerate(idx)}

    words = []
    for start, end, tok in spans:
        j = VOCAB.get(tok)
        if j is None or j not in pos_of:
            continue
        words.append(
            {
                "start": start,
                "end": end,
                "w": [round(float(c), 4) for c in contrib[:, pos_of[j]]],
            }
        )

    top_terms = {}
    for li, label in enumerate(LABELS):
        order = np.argsort(-contrib[li])[:6] if contrib.shape[1] else []
        top_terms[label] = [
            {"term": INV_VOCAB[int(idx[o])], "w": round(float(contrib[li, o]), 4)}
            for o in order
            if contrib[li, o] > 0
        ]

    return {
        "model": body.model,
        "thresholds": thresholds,
        "scores": scores,
        "flagged": flagged,
        "words": words,
        "top_terms": top_terms,
        "chars": len(text),
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
