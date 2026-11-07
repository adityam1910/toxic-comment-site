# Paste this into ONE new Colab cell and run it AFTER the weighted-vs-unweighted
# experiment cells (it needs weighted_model, weighted_thresholds, tokenizer).
import json
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.colab import files

model = weighted_model                # the class-weighted BiLSTM
thresholds = weighted_thresholds      # the validation-tuned thresholds

emb_layer = [l for l in model.layers if isinstance(l, layers.Embedding)][0]
bi = [l for l in model.layers if isinstance(l, layers.Bidirectional)][0]
dense = [l for l in model.layers if isinstance(l, layers.Dense)]
fw = bi.forward_layer.get_weights()
bw = bi.backward_layer.get_weights()

np.savez_compressed(
    "bilstm_weights.npz",
    emb=emb_layer.get_weights()[0],
    fw_k=fw[0], fw_r=fw[1], fw_b=fw[2],
    bw_k=bw[0], bw_r=bw[1], bw_b=bw[2],
    d1_w=dense[0].get_weights()[0], d1_b=dense[0].get_weights()[1],
    d2_w=dense[1].get_weights()[0], d2_b=dense[1].get_weights()[1],
)

# Reference predictions from Keras, so the website can be checked against them
samples = [
    "Thanks for the thoughtful edit, this reads much better now.",
    "You are an idiot and nobody wants you here.",
    "i like your dress",
]
seqs = pad_sequences(
    tokenizer.texts_to_sequences([clean_text(t) for t in samples]),
    maxlen=MAX_LENGTH, padding="post", truncating="post",
)
check = {t: [float(x) for x in p] for t, p in zip(samples, model.predict(seqs, verbose=0))}

meta = {
    "labels": list(LABELS),
    "max_len": int(MAX_LENGTH),
    "num_words": int(MAX_WORDS),
    "oov_index": int(tokenizer.word_index["<OOV>"]),
    "thresholds": [float(t) for t in thresholds],
    "word_index": {w: int(i) for w, i in tokenizer.word_index.items() if i < MAX_WORDS},
    "check": check,
}
with open("bilstm_meta.json", "w") as f:
    json.dump(meta, f)

print("Saved bilstm_weights.npz and bilstm_meta.json")
files.download("bilstm_weights.npz")
files.download("bilstm_meta.json")
