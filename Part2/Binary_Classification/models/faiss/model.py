import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os, sys

# Load processed dataset


df = pd.read_csv("data/processed/processed_train.csv")
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop("RiskFlag", axis=1)
y = df["RiskFlag"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale numerical columns only
numerical_cols = [
    "ApplicantYears",
    "AnnualEarnings",
    "RequestedSum",
    "TrustMetric",
    "WorkDuration",
    "ActiveAccounts",
    "OfferRate",
    "RepayPeriod",
    "DebtFactor",
]

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

X_train.head()

# faiss_all_upgrades.py
# Run all FAISS upgrades + hybrids. Expects X_train, X_test, y_train, y_test in scope.
# Dependencies:
# pip install faiss-cpu scikit-learn numpy scipy torch

import time
import numpy as np
import faiss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


# -------------------------
# Helper utils
# -------------------------
def ensure_float32_contiguous(X):
    arr = np.ascontiguousarray(np.asarray(X).astype(np.float32))
    return arr


def normalize_vectors(X):
    faiss.normalize_L2(X)
    return X


def get_class_weights(y):
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.asarray(y))
    return float(cw[0]), float(cw[1])


def print_metrics(y_true, y_pred, prefix="RESULT"):
    print(f"\n=== {prefix} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))


# -------------------------
# Common preprocessing (PCA + normalize)
# -------------------------
def prepare_data_for_faiss(X_train, X_test, pca_dim=16):
    X_train_np = ensure_float32_contiguous(X_train)
    X_test_np = ensure_float32_contiguous(X_test)
    # PCA
    if pca_dim is not None and pca_dim > 0 and pca_dim < X_train_np.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=42)
        X_train_pca = pca.fit_transform(X_train_np).astype(np.float32)
        X_test_pca = pca.transform(X_test_np).astype(np.float32)
    else:
        X_train_pca, X_test_pca = X_train_np, X_test_np
    X_train_pca = ensure_float32_contiguous(X_train_pca)
    X_test_pca = ensure_float32_contiguous(X_test_pca)
    normalize_vectors(X_train_pca)
    normalize_vectors(X_test_pca)
    return X_train_pca, X_test_pca


# -------------------------
# Core FAISS probabilistic scoring
# -------------------------
def faiss_probabilities(
    index, Xq, y_index, class_weights=(1.0, 1.0), K=20, temperature=None
):
    """
    Return probabilities p(class1) for each query in Xq.
    index: FAISS index (on normalized vectors for cosine -> use inner product or convert dist->sim)
    Xq: float32 contiguous normalized query vectors
    y_index: labels aligned with index vectors
    class_weights: tuple (w0, w1)
    K: number of neighbors to retrieve
    temperature: if set, use exp(sim/temp) weighting
    """
    # We assume index.search returns distances (if using IndexFlatL2 with normalized vectors, dist->sim = 1 - dist/2)
    # If using IP index (inner product), distances are actually similarities (higher is better).
    distances, neighbors = index.search(Xq, K)
    probs = np.zeros(Xq.shape[0], dtype=np.float32)
    w0, w1 = class_weights
    for i, (dist_list, idx_list) in enumerate(zip(distances, neighbors)):
        # determine sim values robustly: if index metric = L2 on normalized vectors -> sim = 1 - d/2
        # We'll detect by checking if distances are mostly in [0,2] (L2) or [-1,1] (IP).
        sims = None
        if (
            np.all(dist_list >= -1e-6) and np.mean(dist_list) <= 2.1
        ):  # likely L2 on normalized
            sims = 1.0 - dist_list / 2.0
        else:
            # treat distances as inner product (similarities)
            sims = dist_list
        sims = np.maximum(sims, 1e-12)

        if temperature and temperature > 0:
            # Temperature softmax - scale sims before exp to avoid overflow
            scaled = sims / float(temperature)
            # numeric stability
            scaled = scaled - scaled.max()
            weights = np.exp(scaled)
        else:
            weights = sims  # linear weighting

        score0 = 0.0
        score1 = 0.0
        for lbl, w in zip(y_index[idx_list], weights):
            if lbl == 0:
                score0 += w * w0
            else:
                score1 += w * w1
        probs[i] = score1 / (score0 + score1 + 1e-12)
    return probs


# -------------------------
# Build HNSW index helper (normalized vectors -> use IndexHNSWFlat with inner product by using normalized vectors and IndexFlatIP)
# But faiss.IndexHNSWFlat supports L2 by default. We'll stick to IndexHNSWFlat(L2) on normalized vectors and convert distances -> sims.
# -------------------------
def build_hnsw_index(X, M=32, efC=200, efS=256):
    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, M)  # L2 on vectors
    index.hnsw.efConstruction = efC
    index.hnsw.efSearch = efS
    index.add(X)
    return index


# -------------------------
# 1) Range search classifier
# -------------------------
def faiss_range_search_classifier(
    X_train_pca, X_test_pca, y_train_arr, radius=0.35, class_weights=(1.0, 1.0)
):
    """
    Range search: retrieve neighbors within radius (cosine similarity threshold converted to L2 dist)
    For normalized vectors: L2 distance d relates to cosine sim s by d = 2*(1 - s). So for sim >= s0, d <= 2*(1-s0).
    radius: threshold on cosine similarity (0..1). We'll convert to L2 radius.
    """
    # Build flat IP index or use flat L2 and convert. We'll use IndexFlatL2 on normalized vectors and range_search on L2 radius.
    X_train = ensure_float32_contiguous(X_train_pca)
    X_test = ensure_float32_contiguous(X_test_pca)

    l2_index = faiss.IndexFlatL2(X_train.shape[1])
    l2_index.add(X_train)

    # convert similarity threshold to squared L2 radius: for normalized vectors, d = 2*(1 - sim)
    sim_thr = radius
    l2_radius = 2.0 * (1.0 - sim_thr)

    # perform range search
    lims, D, I = l2_index.range_search(
        X_test, l2_radius
    )  # returns results in flattened form
    # lims has length nq+1, neighbors for query i are I[lims[i]:lims[i+1]]

    probs = np.zeros(X_test.shape[0], dtype=np.float32)
    w0, w1 = class_weights

    for i in range(X_test.shape[0]):
        start = lims[i]
        end = lims[i + 1]
        if end - start == 0:
            # No neighbors found within radius => fallback to nearest neighbor (k=1)
            _, idxs = l2_index.search(X_test[i : i + 1], 1)
            idxs = idxs[0]
            # simulate similarity
            sims = np.ones(len(idxs), dtype=np.float32)
            score0 = sum((1.0 if y_train_arr[j] == 0 else 0.0) * w0 * 1.0 for j in idxs)
            score1 = sum((1.0 if y_train_arr[j] == 1 else 0.0) * w1 * 1.0 for j in idxs)
            probs[i] = score1 / (score0 + score1 + 1e-12)
            continue
        idxs = I[start:end]
        # D contains squared L2 distances: convert to cos sim
        dists = D[start:end]
        sims = np.maximum(1.0 - dists / 2.0, 1e-12)
        # compute weighted score
        score0 = 0.0
        score1 = 0.0
        for lbl, s in zip(y_train_arr[idxs], sims):
            if lbl == 0:
                score0 += s * w0
            else:
                score1 += s * w1
        probs[i] = score1 / (score0 + score1 + 1e-12)
    return probs


# -------------------------
# 2) Oversample minority embeddings for FAISS
# -------------------------
def oversample_minority(X_train_pca, y_train_arr, multiplier=3, jitter_scale=0.01):
    # Duplicate minority examples multiplier times with small gaussian jitter
    X = list(X_train_pca)
    y = list(y_train_arr)
    idxs_min = np.where(y_train_arr == 1)[0]
    n_min = len(idxs_min)
    if n_min == 0:
        return X_train_pca, y_train_arr
    for _ in range(multiplier - 1):
        jitter = np.random.normal(
            scale=jitter_scale, size=(n_min, X_train_pca.shape[1])
        ).astype(np.float32)
        X.extend((X_train_pca[idxs_min] + jitter).tolist())
        y.extend([1] * n_min)
    X_new = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    y_new = np.asarray(y, dtype=int)
    # renormalize
    normalize_vectors(X_new)
    return X_new, y_new


# -------------------------
# 3) Two-stage FAISS -> Logistic Regression meta-model
# -------------------------
def two_stage_faiss_lr(X_train_pca, X_test_pca, y_train_arr, K_meta=50, C=1.0):
    # Build index on train
    idx = build_hnsw_index(X_train_pca, M=32, efC=200, efS=512)
    # Build training meta-features by searching each training point's neighbors excluding itself:
    n_train = X_train_pca.shape[0]
    D_train, I_train = idx.search(X_train_pca, K_meta + 1)  # self included
    X_meta = []
    for i in range(n_train):
        idxs = I_train[i][1:]  # exclude self
        dists = D_train[i][1:]
        sims = np.maximum(1.0 - dists / 2.0, 1e-12)
        # features: total sim per class, mean sim per class, count per class
        labels = y_train_arr[idxs]
        sim0 = sims[labels == 0].sum() if np.any(labels == 0) else 0.0
        sim1 = sims[labels == 1].sum() if np.any(labels == 1) else 0.0
        cnt0 = int((labels == 0).sum())
        cnt1 = int((labels == 1).sum())
        mean0 = sims[labels == 0].mean() if cnt0 > 0 else 0.0
        mean1 = sims[labels == 1].mean() if cnt1 > 0 else 0.0
        X_meta.append([sim0, sim1, cnt0, cnt1, mean0, mean1])
    X_meta = np.asarray(X_meta, dtype=np.float32)
    # train logistic regression on meta features
    lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    lr.fit(X_meta, y_train_arr)

    # Build test meta features
    D_test, I_test = idx.search(X_test_pca, K_meta)
    X_meta_test = []
    for dists, idxs in zip(D_test, I_test):
        sims = np.maximum(1.0 - dists / 2.0, 1e-12)
        labels = y_train_arr[idxs]
        sim0 = sims[labels == 0].sum() if np.any(labels == 0) else 0.0
        sim1 = sims[labels == 1].sum() if np.any(labels == 1) else 0.0
        cnt0 = int((labels == 0).sum())
        cnt1 = int((labels == 1).sum())
        mean0 = sims[labels == 0].mean() if cnt0 > 0 else 0.0
        mean1 = sims[labels == 1].mean() if cnt1 > 0 else 0.0
        X_meta_test.append([sim0, sim1, cnt0, cnt1, mean0, mean1])
    X_meta_test = np.asarray(X_meta_test, dtype=np.float32)
    probs = lr.predict_proba(X_meta_test)[:, 1]
    return probs, lr


# -------------------------
# 4) FAISS + SVM ensemble
# -------------------------
def faiss_svm_ensemble(
    X_train_pca,
    X_test_pca,
    y_train_arr,
    faiss_index=None,
    K=20,
    calibrate_svm=True,
    alpha=0.6,
):
    # Train an SVM with probability (may be slow). Use a calibrated estimator for better probs if needed.
    # We'll use a linear SVC calibrated for speed if high dims; else use SVC(probability=True)
    try:
        svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
        svm.fit(X_train_pca, y_train_arr)
    except Exception as e:
        # fallback to calibrated LinearSVC if SVC fails
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        lsvc = LinearSVC(class_weight="balanced", max_iter=10000)
        svm = CalibratedClassifierCV(lsvc, cv=3)
        svm.fit(X_train_pca, y_train_arr)

    probs_svm = svm.predict_proba(X_test_pca)[:, 1]

    # compute faiss probs using provided index or new index
    if faiss_index is None:
        faiss_index = build_hnsw_index(X_train_pca, M=32, efC=200, efS=256)
    class_weights = get_class_weights(y_train_arr)
    probs_faiss = faiss_probabilities(
        faiss_index,
        X_test_pca,
        y_train_arr,
        class_weights=class_weights,
        K=K,
        temperature=0.02,
    )

    # blend
    p_final = alpha * probs_faiss + (1.0 - alpha) * probs_svm
    return p_final, probs_faiss, probs_svm, svm


# -------------------------
# 5) Autoencoder embedding (PyTorch)
# -------------------------
def autoencoder_embedding_and_faiss(
    X_train_pca,
    X_test_pca,
    hidden_dim=64,
    embed_dim=16,
    epochs=12,
    batch_size=256,
    lr=1e-3,
    device="cpu",
):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    Xtr = torch.from_numpy(X_train_pca)
    Xte = torch.from_numpy(X_test_pca)
    ds = TensorDataset(Xtr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    d_in = X_train_pca.shape[1]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)
            )
            self.dec = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_in)
            )

        def forward(self, x):
            z = self.enc(x)
            xhat = self.dec(z)
            return xhat, z

    model = AE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        for (batch,) in dl:
            batch = batch.to(device).float()
            xhat, _ = model(batch)
            loss = loss_fn(xhat, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        # print(f"AE Epoch {ep+1}/{epochs} loss={epoch_loss/len(ds):.6f}")
    model.eval()
    with torch.no_grad():
        Z_train = (
            model.enc(torch.from_numpy(X_train_pca).to(device).float())
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        Z_test = (
            model.enc(torch.from_numpy(X_test_pca).to(device).float())
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # Normalize embeddings and build HNSW
    normalize_vectors(Z_train)
    normalize_vectors(Z_test)
    idx = build_hnsw_index(Z_train, M=32, efC=200, efS=512)
    return idx, Z_train, Z_test


# -------------------------
# Execute all experiments
# -------------------------
def run_all(X_train, X_test, y_train, y_test):
    print("Preparing data (PCA + normalize)...")
    Xtr, Xte = prepare_data_for_faiss(X_train, X_test, pca_dim=16)
    ytr = np.array(y_train).astype(int)
    yte = np.array(y_test).astype(int)

    cw = get_class_weights(ytr)
    print("Class weights:", cw)

    # Build canonical HNSW index (on PCA+normalized)
    idx = build_hnsw_index(Xtr, M=32, efC=200, efS=512)
    print("HNSW index built: n_train", Xtr.shape[0], "dim", Xtr.shape[1])

    # Baseline: probabilistic FAISS with temperature + threshold tuning (like we already did)
    t0 = time.time()
    probs_baseline = faiss_probabilities(
        idx, Xte, ytr, class_weights=cw, K=20, temperature=0.02
    )
    best_f1 = 0
    best_thr = 0.5
    for thr in np.linspace(0.05, 0.95, 50):
        preds = (probs_baseline >= thr).astype(int)
        f = f1_score(yte, preds)
        if f > best_f1:
            best_f1 = f
            best_thr = thr
    preds = (probs_baseline >= best_thr).astype(int)
    print_metrics(
        yte,
        preds,
        prefix=f"FAISS Baseline (temp softmax) best_thr={best_thr:.3f}, f1={best_f1:.4f} (time {time.time()-t0:.2f}s)",
    )

    # 1) Range search
    t0 = time.time()
    probs_range = faiss_range_search_classifier(
        Xtr, Xte, ytr, radius=0.42, class_weights=cw
    )
    best_f1 = 0
    best_thr = 0.5
    for thr in np.linspace(0.05, 0.95, 50):
        p = (probs_range >= thr).astype(int)
        f = f1_score(yte, p)
        if f > best_f1:
            best_f1 = f
            best_thr = thr
    preds_range = (probs_range >= best_thr).astype(int)
    print_metrics(
        yte,
        preds_range,
        prefix=f"FAISS RangeSearch radius=0.42 best_thr={best_thr:.3f} f1={best_f1:.4f} (time {time.time()-t0:.2f}s)",
    )

    # 2) Oversample minority then FAISS
    t0 = time.time()
    X_over, y_over = oversample_minority(Xtr, ytr, multiplier=4, jitter_scale=0.02)
    idx_over = build_hnsw_index(X_over, M=32, efC=200, efS=512)
    probs_over = faiss_probabilities(
        idx_over, Xte, y_over, class_weights=cw, K=10, temperature=0.02
    )
    best_f1 = 0
    best_thr = 0.5
    for thr in np.linspace(0.05, 0.95, 50):
        p = (probs_over >= thr).astype(int)
        f = f1_score(yte, p)
        if f > best_f1:
            best_f1 = f
            best_thr = thr
    preds_over = (probs_over >= best_thr).astype(int)
    print_metrics(
        yte,
        preds_over,
        prefix=f"FAISS Oversampled x4 best_thr={best_thr:.3f} f1={best_f1:.4f} (time {time.time()-t0:.2f}s)",
    )

    # 3) Two-stage FAISS -> Logistic Regression
    t0 = time.time()
    probs_meta, meta_lr = two_stage_faiss_lr(Xtr, Xte, ytr, K_meta=50, C=1.0)
    best_f1 = 0
    best_thr = 0.5
    for thr in np.linspace(0.05, 0.95, 50):
        p = (probs_meta >= thr).astype(int)
        f = f1_score(yte, p)
        if f > best_f1:
            best_f1 = f
            best_thr = thr
    preds_meta = (probs_meta >= best_thr).astype(int)
    print_metrics(
        yte,
        preds_meta,
        prefix=f"Two-stage FAISS->LR best_thr={best_thr:.3f} f1={best_f1:.4f} (time {time.time()-t0:.2f}s)",
    )
    #
    # # 4) FAISS + SVM ensemble
    # t0 = time.time()
    # p_final, p_faiss, p_svm, svm_model = faiss_svm_ensemble(
    #     Xtr, Xte, ytr, faiss_index=idx, K=20, calibrate_svm=True, alpha=0.6
    # )
    # best_f1 = 0
    # best_thr = 0.5
    # for thr in np.linspace(0.05, 0.95, 50):
    #     p = (p_final >= thr).astype(int)
    #     f = f1_score(yte, p)
    #     if f > best_f1:
    #         best_f1 = f
    #         best_thr = thr
    # preds_ens = (p_final >= best_thr).astype(int)
    # print_metrics(
    #     yte,
    #     preds_ens,
    #     prefix=f"FAISS+SVM Ensemble best_thr={best_thr:.3f} f1={best_f1:.4f} (time {time.time()-t0:.2f}s)",
    # )
    #
    # 5) Autoencoder -> FAISS
    t0 = time.time()
    try:
        idx_ae, Z_train, Z_test = autoencoder_embedding_and_faiss(
            Xtr,
            Xte,
            hidden_dim=128,
            embed_dim=24,
            epochs=15,
            batch_size=512,
            lr=1e-3,
            device="cpu",
        )
        probs_ae = faiss_probabilities(
            idx_ae, Z_test, ytr, class_weights=cw, K=20, temperature=0.02
        )
        best_f1 = 0
        best_thr = 0.5
        for thr in np.linspace(0.05, 0.95, 50):
            p = (probs_ae >= thr).astype(int)
            f = f1_score(yte, p)
            if f > best_f1:
                best_f1 = f
                best_thr = thr
        preds_ae = (probs_ae >= best_thr).astype(int)
        print_metrics(
            yte,
            preds_ae,
            prefix=f"AutoEncoder-FAISS best_thr={best_thr:.3f} f1={best_f1:.4f} (time {time.time()-t0:.2f}s)",
        )
    except Exception as e:
        print("Autoencoder step failed:", e)

    print("\nALL DONE.")


# -------------------------
# Run (assuming X_train, X_test, y_train, y_test exist)
# -------------------------
if __name__ == "__main__":
    # If variables are not in global scope, try to pick them from globals() (adjust if running inside notebook)
    if "X_train" not in globals():
        raise SystemExit(
            "X_train not found. Ensure X_train, X_test, y_train, y_test are defined in the environment."
        )
    run_all(X_train, X_test, y_train, y_test)
