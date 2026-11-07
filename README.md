# Toxic Comment Classifier

A full-stack web app that scores a comment on six toxicity categories (toxic, severe toxic, obscene, threat, insult, identity hate) and highlights the words that drove each score.

- **Backend:** FastAPI (`app/main.py`), serves the API and the website
- **Frontend:** plain HTML, CSS and JavaScript (`app/static/`)
- **Models:** TF-IDF + Logistic Regression and Complement Naive Bayes, trained on the Jigsaw Toxic Comment dataset (`models/`)
- **Optional BiLSTM:** add `bilstm_weights.npz` and `bilstm_meta.json` to `models/` (exported from Colab with `export_bilstm_for_website.py`) and a third model, "BiLSTM (weighted)", appears automatically. It runs in NumPy, so TensorFlow is not needed on the server.

Comments are scored in memory and are not stored.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000. API docs are at http://localhost:8000/api/docs.
Live at https://toxic-comment-site.onrender.com/

## API

`POST /api/predict`

```json
{ "text": "i like your dress", "model": "lr" }
```

`model` is `lr`, `nb`, or `bilstm` (only when its files are present; `GET /api/health` lists what is available). The response has `scores` (six probabilities), `thresholds` (decision threshold per label), `flagged` (labels at or above their threshold), `words` (per-word weights used for highlighting) and `top_terms`.

## Deploy on Render (free)

1. Push this folder to a GitHub repository.
2. On render.com choose New, then Blueprint (or Web Service) and connect the repository.
3. Render reads `render.yaml`. Build command: `pip install -r requirements.txt`. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
4. Open the `.onrender.com` link once the deploy finishes.

The free plan sleeps when idle, so the first request after a break can take up to a minute.

## Notes

`scikit-learn` is pinned to 1.6.1 because the `.joblib` files were saved with that version. Do not upgrade it without re-saving the models.

Student research project, Manipal University Jaipur. Not a moderation product.
