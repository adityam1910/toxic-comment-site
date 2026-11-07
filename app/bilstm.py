"""NumPy-only inference for the weighted BiLSTM trained in Colab.

Architecture (matches the notebook's build_bilstm):
Embedding(50000, 128, mask_zero=True) -> Bidirectional(LSTM(64)) ->
Dense(64, relu) -> Dropout -> Dense(6, sigmoid)

Running it in NumPy avoids installing TensorFlow (about 500 MB), which would not
fit on a free hosting plan. Dropout is inactive at inference time.
"""
import json
import re
import string
from pathlib import Path

import numpy as np

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Exact copy of the notebook's clean_text() used to train the BiLSTM."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _run_lstm(ids_emb, kernel, recurrent, bias, units):
    """Run one LSTM over a (T, D) sequence and return the final hidden state.
    Keras gate order is input, forget, cell, output."""
    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)
    for x in ids_emb:
        z = x @ kernel + h @ recurrent + bias
        i = _sigmoid(z[:units])
        f = _sigmoid(z[units : 2 * units])
        g = np.tanh(z[2 * units : 3 * units])
        o = _sigmoid(z[3 * units :])
        c = f * c + i * g
        h = o * np.tanh(c)
    return h


class BiLSTM:
    def __init__(self, weights_path: Path, meta_path: Path):
        w = np.load(weights_path)
        self.emb = w["emb"].astype(np.float32)
        self.fw = (w["fw_k"], w["fw_r"], w["fw_b"])
        self.bw = (w["bw_k"], w["bw_r"], w["bw_b"])
        self.d1 = (w["d1_w"], w["d1_b"])
        self.d2 = (w["d2_w"], w["d2_b"])
        self.units = self.fw[1].shape[0]

        meta = json.loads(Path(meta_path).read_text())
        self.labels = meta["labels"]
        self.max_len = int(meta["max_len"])
        self.num_words = int(meta["num_words"])
        self.oov = int(meta["oov_index"])
        self.word_index = meta["word_index"]
        self.thresholds = {l: float(t) for l, t in zip(self.labels, meta["thresholds"])}

    def encode(self, cleaned: str):
        """Same rules as Keras Tokenizer.texts_to_sequences with an OOV token,
        then truncation to max_len (post)."""
        ids = []
        for word in cleaned.split(" "):
            if not word:
                continue
            i = self.word_index.get(word)
            ids.append(i if (i is not None and i < self.num_words) else self.oov)
        return ids[: self.max_len]

    def predict_proba(self, text: str):
        ids = self.encode(clean_text(text))
        if not ids:
            h = np.zeros(2 * self.units, dtype=np.float32)
        else:
            seq = self.emb[ids]
            h_f = _run_lstm(seq, *self.fw, self.units)
            h_b = _run_lstm(seq[::-1], *self.bw, self.units)
            h = np.concatenate([h_f, h_b])
        a = np.maximum(h @ self.d1[0] + self.d1[1], 0.0)
        return _sigmoid(a @ self.d2[0] + self.d2[1])
