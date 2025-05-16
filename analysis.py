import re, json, collections
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models.fasttext import FastText
from sklearn.manifold import TSNE
import umap

TEXT_DIR   = Path("./preprocessed")          # decade .txt files (tokenised)
MODEL_DIR  = Path("./embedding_models")      # fastText_<decade>.model
LOAN_CSV   = Path("./turkish_farsi_words.csv")
OUT_DIR    = Path("./analysis_outputs")
ANCHOR_TOP = 5000        # top-freq words for alignment anchors
TOP_N_PLOT = 20          # how many loan-words to plot in frequency chart
SEED       = 42

sns.set_style("whitegrid")
OUT_DIR.mkdir(exist_ok=True, parents=True)


print("Loading decade corpora & models …")
decade_paths = sorted(TEXT_DIR.glob("preprocessed_*.txt"))
decades      = [re.search(r"(\d{4})", p.name).group(1) for p in decade_paths]


models = {
    d: FastText.load(MODEL_DIR / f"fasttext_{d}.model").wv
    for d in decades
}

loan_df = pd.read_csv(LOAN_CSV)           # cols: Turkish,Farsi
loanwords = loan_df["Turkish"].str.strip().tolist()


def align_to_reference(ref_kv, kv_to_align, anchors):
    X = np.stack([ref_kv[w] for w in anchors])
    Y = np.stack([kv_to_align[w] for w in anchors])
    R, _ = orthogonal_procrustes(Y, X)      # Y·R ≈ X
    aligned_vecs = kv_to_align.get_normed_vectors(copy=False) @ R
    new_kv = kv_to_align.copy()
    new_kv.fill_norms = False
    new_kv.vectors = aligned_vecs
    return new_kv

ref_decade = decades[0]
ref_kv     = models[ref_decade]
anchor_words = ref_kv.index_to_key[:ANCHOR_TOP]

aligned = {ref_decade: ref_kv}
for d in decades[1:]:
    aligned[d] = align_to_reference(ref_kv, models[d], anchor_words)
print("All decades aligned to reference space.")


def freq_for_decade(txt_path):
    counts = collections.Counter(Path(txt_path).read_text(encoding="utf-8").split())
    total  = sum(counts.values())
    return {w: counts[w] / total for w in loanwords}

freq_df = pd.DataFrame({
    d: freq_for_decade(p) for d, p in zip(decades, decade_paths)
}).T.fillna(0.0)   # shape: decades × loanwords
freq_df.to_csv(OUT_DIR / "loanword_frequency.csv", index_label="decade")

# Plot top-N loan-words by max frequency
top_loans = freq_df.max().sort_values(ascending=False).head(TOP_N_PLOT).index
plt.figure(figsize=(11,4))
for w in top_loans:
    plt.plot(freq_df.index, freq_df[w], marker='o', label=w)
plt.yscale("log")
plt.xlabel("Decade")
plt.ylabel("Relative frequency (log scale)")
plt.title("Persian loan-word frequency over time")
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "frequency_lines.png", dpi=300)
plt.close()


def cosine_shift(word, kv_a, kv_b):
    return 1 - np.dot(kv_a[word], kv_b[word])

first_kv, last_kv = aligned[decades[0]], aligned[decades[-1]]
cos_shift = {w: cosine_shift(w, first_kv, last_kv)
             for w in loanwords if w in first_kv and w in last_kv}

# Histogram
plt.figure(figsize=(6,4))
sns.histplot(list(cos_shift.values()), bins=25, kde=True)
plt.xlabel("Cosine distance (earliest → latest)")
plt.title("Semantic drift of Persian loan-words")
plt.tight_layout()
plt.savefig(OUT_DIR / "shift_histogram.png", dpi=300)
plt.close()

# Neighbour overlap @k
def neighbours(kv, word, k=10):
    return {w for w, _ in kv.most_similar(word, topn=k)}

k = 10
nbr_overlap = {
    w: 1 - len(neighbours(first_kv, w, k) & neighbours(last_kv, w, k)) / k
    for w in cos_shift
}

pd.DataFrame({
    "cosine_shift": cos_shift,
    f"nbr_overlap@{k}": nbr_overlap
}).to_csv(OUT_DIR / "semantic_metrics.csv", index_label="word")


print("Running UMAP for 2-D visualisation …")
rng = np.random.RandomState(SEED)
embed_mat = np.stack([last_kv[w] for w in loanwords if w in last_kv])
labels    = [w for w in loanwords if w in last_kv]

umap_2d = umap.UMAP(n_neighbors=50, min_dist=0.3, metric="cosine",
                    random_state=rng).fit_transform(embed_mat)

plt.figure(figsize=(7,6))
plt.scatter(umap_2d[:,0], umap_2d[:,1], s=18, alpha=0.7)
for (x,y),w in zip(umap_2d, labels):
    plt.text(x, y, w, fontsize=7, alpha=0.9)
plt.title(f"UMAP of loan-word vectors ({decades[-1]})")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "umap_loanwords.png", dpi=300)
plt.close()


summary = {
    "decades": decades,
    "anchor_words": ANCHOR_TOP,
    "loanwords_total": len(loanwords),
    "coverage_reference": sum(w in ref_kv for w in loanwords) / len(loanwords),
    "files_written": [str(p) for p in OUT_DIR.glob("*.*")]
}
(Path(OUT_DIR) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Analysis complete! Outputs are in", OUT_DIR)
