import csv, numpy as np
from pathlib import Path
from scipy import spatial   # already required later


_DICT_CSV = Path("turkish_farsi_words.csv")   # adjust if elsewhere

def _lazy_load_csv():
    tr_words, fa_words = [], []
    with open(_DICT_CSV, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            tr_words.append(row["Turkish"].strip())
            fa_words.append(row["Farsi"].strip())
    return tr_words, fa_words


def eval_loanword_coverage(model) -> float:
    tr_words, _ = _lazy_load_csv()
    covered = sum(1 for w in tr_words if w in model.wv.key_to_index)
    return covered / len(tr_words)


def eval_alignment(model, top_k: int = 1) -> float:
    tr_words, fa_words = _lazy_load_csv()
    pairs = [(tr, fa) for tr, fa in zip(tr_words, fa_words)
             if tr in model.wv.key_to_index and fa in model.wv.key_to_index]
    if not pairs:
        return 0.0

    vecs = model.wv.get_normed_vectors()
    index  = {w: i for i, w in enumerate(model.wv.index_to_key)}

    correct = 0
    for tr, fa in pairs:
        v_tr = vecs[index[tr]]
        sims = np.dot(vecs, v_tr) # cosine sims to all vocab
        nn_ids = sims.argsort()[-top_k:][::-1]
        nn_words = {model.wv.index_to_key[i] for i in nn_ids}
        correct += fa in nn_words

    return correct / len(pairs)
