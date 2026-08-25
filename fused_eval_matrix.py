"""
fused_eval_matrix.py — Comprehensive Multimodal Fused Embeddings Evaluation Suite
==================================================================================

Performs exhaustive biometric evaluation of multimodal fused embeddings
(e.g., CBP Compact Bilinear Pooling or 1536-D Concat-L2 representations):

    1. Distance & Similarity Measures:
        - Cosine similarity
        - Euclidean distance
        - Manhattan (L1) distance
        - Hamming distance (sign-binarized)

    2. Verification Accuracy & Discrimination Metrics:
        - Overall Classification Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Equal Error Rate (EER) + optimal EER threshold
        - False Accept Rate (FAR) & False Reject Rate (FRR)
        - Genuine Accept Rate (GAR) & True Accept Rate (TAR)
        - Decidability Index (d-prime separation)
        - TAR at fixed security operating points: FAR @ 1%, 0.1%, 0.01%
        - Person-level subject bootstrap 95% Confidence Intervals for EER & Accuracy
        - Confusion Matrix (TP, FP, TN, FN counts & rates at EER threshold)
        - Pairwise Decision Matrix (TP / FN / FP / TN per person pair)
        - Per-person local thresholds, local EER, local accuracy
        - Threshold Journey Matrix (1000 threshold steps x N persons)

    3. Visualizations & Graphs (saved as PNG with accompanying CSV data tables):
        - DET Curve (FAR vs FRR, log-log)
        - ROC Curve (FAR vs TAR)
        - Genuine vs Impostor Score Distribution Histogram
        - Confusion Matrix Heatmap (2x2 annotated)
        - Decision Matrix Heatmap (per-person pair colored)
        - Bootstrap EER Distribution Histogram

Usage:
    python fused_eval_matrix.py --db database/biometric.db --dataset setA --fusion_type cbp --out_dir results/fused_eval
    python fused_eval_matrix.py --db database/biometric.db --dataset setA --fusion_type concat_l2 --template_type raw --out_dir results/fused_concat
"""

import argparse
import os
import sqlite3
from typing import Optional, Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================================
# 1. LOAD FUSED DATA FROM SQLITE
# ==========================================================================

def load_fused_data(
    db_path: str,
    dataset: str,
    fusion_type: str = "cbp",
    template_type: str = "raw",
) -> Dict[str, Dict[str, any]]:
    """
    Loads fused enrollment templates and live/probe embeddings from SQLite.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    if template_type == "biohash":
        tmpl_table = "biohash_embedding_fused" if "biohash_embedding_fused" in tables else "biohash_fused"
        live_table = "biohashed_livefused" if "biohashed_livefused" in tables else "biohash_live_fused"
    else:
        tmpl_table = "fused_templates" if "fused_templates" in tables else "templates"
        live_table = "fused_live_embeddings" if "fused_live_embeddings" in tables else (
            "fused_image_embeddings" if "fused_image_embeddings" in tables else "live_embeddings"
        )

    if tmpl_table not in tables:
        conn.close()
        raise ValueError(
            f"Fused template table '{tmpl_table}' not found in {db_path}. Available tables: {tables}"
        )

    # Check if fusion_type column exists in tmpl_table
    tmpl_cols = [c[1] for c in cur.execute(f"PRAGMA table_info({tmpl_table})").fetchall()]
    if "fusion_type" in tmpl_cols:
        tmpl_query = f"SELECT person_id, embedding, dim FROM {tmpl_table} WHERE dataset=? AND fusion_type=? ORDER BY person_id"
        tmpl_params = (dataset, fusion_type)
    else:
        tmpl_query = f"SELECT person_id, embedding, dim FROM {tmpl_table} WHERE dataset=? ORDER BY person_id"
        tmpl_params = (dataset,)

    tmpl_rows = cur.execute(tmpl_query, tmpl_params).fetchall()

    live_rows = []
    if live_table in tables:
        live_cols = [c[1] for c in cur.execute(f"PRAGMA table_info({live_table})").fetchall()]
        idx_col = "live_index" if "live_index" in live_cols else ("probe_index" if "probe_index" in live_cols else "image_index")
        
        if "fusion_type" in live_cols:
            live_query = f"SELECT person_id, {idx_col}, embedding, dim FROM {live_table} WHERE dataset=? AND fusion_type=? ORDER BY person_id, {idx_col}"
            live_params = (dataset, fusion_type)
        else:
            live_query = f"SELECT person_id, {idx_col}, embedding, dim FROM {live_table} WHERE dataset=? ORDER BY person_id, {idx_col}"
            live_params = (dataset,)
            
        live_rows = cur.execute(live_query, live_params).fetchall()
    else:
        print(f"[WARN] Table '{live_table}' not found in {db_path}. Enrollment templates will be used for evaluation.")

    conn.close()

    if not tmpl_rows:
        raise ValueError(
            f"No fused templates found in '{tmpl_table}' for dataset='{dataset}', fusion_type='{fusion_type}', template_type='{template_type}'."
        )

    data = {}
    for pid, blob, dim in tmpl_rows:
        emb = np.frombuffer(blob, dtype=np.float32).astype(np.float64).copy()
        data.setdefault(pid, {})["enrollment"] = emb
        data[pid]["live_sets"] = []

    for pid, l_idx, blob, dim in live_rows:
        if pid in data:
            emb = np.frombuffer(blob, dtype=np.float32).astype(np.float64).copy()
            data[pid]["live_sets"].append((l_idx, emb))

    print(f"[DB] Loaded {len(data)} persons from '{tmpl_table}' & '{live_table}'  fusion_type={fusion_type!r}  type={template_type.upper()}")
    n_live = sum(len(entry["live_sets"]) for entry in data.values())
    print(f"     Total live set embeddings loaded across all persons: {n_live}")

    return data


# ==========================================================================
# 2. SIMILARITY / DISTANCE METRICS
# ==========================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_n, b_n))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Hamming distance in [0, 1] (fraction of differing bits)."""
    a_bin = np.sign(a)
    b_bin = np.sign(b)
    return float(np.mean(a_bin != b_bin))


METRICS = {
    "cosine":    {"func": cosine_similarity,  "higher_is_match": True,  "label": "Cosine Similarity"},
    "euclidean": {"func": euclidean_distance, "higher_is_match": False, "label": "Euclidean Distance"},
    "manhattan": {"func": manhattan_distance, "higher_is_match": False, "label": "Manhattan Distance"},
    "hamming":   {"func": hamming_distance,   "higher_is_match": False, "label": "Hamming Distance"},
}


# ==========================================================================
# 3. SCORE MATRIX & GENUINE/IMPOSTOR EXTRACTION (per metric)
# ==========================================================================

def compute_fused_scores_and_matrices(person_ids: List[str], data: Dict[str, Any], metric_name: str):
    """
    Computes full person x person score matrix, flat genuine/impostor arrays,
    and grouped per-person scores for subject-level bootstrap.
    """
    metric_fn = METRICS[metric_name]["func"]

    enrollment = {pid: data[pid]["enrollment"] for pid in person_ids}
    live_sets  = {pid: [emb for _, emb in data[pid]["live_sets"]] for pid in person_ids}

    n = len(person_ids)
    score_mat = np.zeros((n, n))

    gen_scores, imp_scores = [], []
    gen_by_person = {pid: [] for pid in person_ids}
    imp_by_person = {pid: [] for pid in person_ids}

    for i, pi in enumerate(person_ids):
        samples = live_sets[pi] if live_sets[pi] else [enrollment[pi]]

        for j, pj in enumerate(person_ids):
            e_vec = enrollment[pj]
            pair_scores = [metric_fn(s, e_vec) for s in samples]
            score_mat[i, j] = float(np.mean(pair_scores))

            if i == j:
                gen_scores.extend(pair_scores)
                gen_by_person[pi].extend(pair_scores)
            else:
                imp_scores.extend(pair_scores)
                imp_by_person[pi].extend(pair_scores)

    score_df = pd.DataFrame(score_mat, index=person_ids, columns=person_ids)

    return (
        np.array(gen_scores), np.array(imp_scores),
        gen_by_person, imp_by_person,
        score_df
    )


def genuine_impostor_label_matrix(person_ids: List[str]) -> pd.DataFrame:
    n = len(person_ids)
    labels = np.eye(n, dtype=bool)
    return pd.DataFrame(labels, index=person_ids, columns=person_ids)


# ==========================================================================
# 4. THRESHOLD SWEEP, EER, ACCURACY, GAR/TAR, D-PRIME
# ==========================================================================

def threshold_sweep(genuine_scores: np.ndarray, impostor_scores: np.ndarray, higher_is_match: bool = True, num_thresholds: int = 1000) -> pd.DataFrame:
    """
    Full threshold sweep with FAR, FRR, GAR, TAR, and overall Classification Accuracy:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)
    n_gen, n_imp = len(genuine_scores), len(impostor_scores)
    total_pairs = n_gen + n_imp

    far_list, frr_list = [], []
    tp_list, tn_list, fp_list, fn_list = [], [], [], []
    acc_list = []

    for t in thresholds:
        if higher_is_match:
            tp = int(np.sum(genuine_scores >= t))
            fn = int(np.sum(genuine_scores < t))
            fp = int(np.sum(impostor_scores >= t))
            tn = int(np.sum(impostor_scores < t))
        else:
            tp = int(np.sum(genuine_scores <= t))
            fn = int(np.sum(genuine_scores > t))
            fp = int(np.sum(impostor_scores <= t))
            tn = int(np.sum(impostor_scores > t))

        far = fp / n_imp if n_imp else 0.0
        frr = fn / n_gen if n_gen else 0.0
        acc = (tp + tn) / max(total_pairs, 1)

        far_list.append(far)
        frr_list.append(frr)
        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        acc_list.append(acc)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    acc_arr = np.array(acc_list)

    return pd.DataFrame({
        "threshold": thresholds,
        "FAR":       far_arr,
        "FRR":       frr_arr,
        "GAR":       1.0 - frr_arr,
        "TAR":       1.0 - frr_arr,
        "accuracy":  acc_arr,
        "TP":        tp_list,
        "TN":        tn_list,
        "FP":        fp_list,
        "FN":        fn_list,
        "abs_diff":  np.abs(far_arr - frr_arr),
    })


def compute_eer(sweep_df: pd.DataFrame) -> Tuple[float, float]:
    idx = sweep_df["abs_diff"].idxmin()
    row = sweep_df.loc[idx]
    return float((row["FAR"] + row["FRR"]) / 2.0), float(row["threshold"])


def tar_at_far_targets(sweep_df: pd.DataFrame, far_targets=(0.01, 0.001, 0.0001), higher_is_match: bool = True) -> pd.DataFrame:
    rows = []
    df_sorted = sweep_df.sort_values("threshold", ascending=not higher_is_match).reset_index(drop=True)

    for target in far_targets:
        candidates = df_sorted[df_sorted["FAR"] <= target]
        if len(candidates) > 0:
            row = candidates.iloc[candidates["FAR"].values.argmax()]
            achieved = True
        else:
            row = df_sorted.iloc[(df_sorted["FAR"] - target).abs().values.argmin()]
            achieved = False

        rows.append({
            "target_FAR":     target,
            "achieved_FAR":   float(row["FAR"]),
            "FAR_target_met": achieved,
            "TAR":            float(row["TAR"]),
            "FRR":            float(row["FRR"]),
            "threshold":      float(row["threshold"]),
        })

    return pd.DataFrame(rows)


def d_prime(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """Decidability index / d-prime."""
    mu_g, mu_i = genuine_scores.mean(), impostor_scores.mean()
    var_g, var_i = genuine_scores.var(), impostor_scores.var()
    denom = np.sqrt((var_g + var_i) / 2.0)
    if denom < 1e-12:
        return 0.0
    return float(abs(mu_g - mu_i) / denom)


# ==========================================================================
# 5. PERSON-LEVEL BOOTSTRAP CONFIDENCE INTERVALS
# ==========================================================================

def bootstrap_eer_person_level(
    person_ids: List[str],
    gen_by_person: Dict[str, list],
    imp_by_person: Dict[str, list],
    higher_is_match: bool = True,
    n_bootstrap: int = 500,
    ci: int = 95,
    num_thresholds: int = 300,
    seed: int = 42,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Person-level bootstrap for EER and Accuracy confidence intervals.
    """
    rng = np.random.RandomState(seed)
    eer_samples = []
    acc_samples = []
    n_persons = len(person_ids)

    for _ in range(n_bootstrap):
        resampled_persons = rng.choice(person_ids, size=n_persons, replace=True)

        gen_pool, imp_pool = [], []
        for pid in resampled_persons:
            gen_pool.extend(gen_by_person[pid])
            imp_pool.extend(imp_by_person[pid])

        if len(gen_pool) == 0 or len(imp_pool) == 0:
            continue

        sweep_df = threshold_sweep(
            np.array(gen_pool), np.array(imp_pool),
            higher_is_match=higher_is_match, num_thresholds=num_thresholds
        )
        eer, eer_t = compute_eer(sweep_df)
        eer_samples.append(eer)
        
        # Accuracy at EER threshold
        best_idx = sweep_df["abs_diff"].idxmin()
        acc_samples.append(sweep_df.loc[best_idx, "accuracy"])

    eer_samples = np.array(eer_samples)
    acc_samples = np.array(acc_samples)
    
    lower = np.percentile(eer_samples, (100 - ci) / 2)
    upper = np.percentile(eer_samples, 100 - (100 - ci) / 2)

    return {
        "eer_bootstrap_mean": float(eer_samples.mean()),
        "eer_bootstrap_std":  float(eer_samples.std()),
        "acc_bootstrap_mean": float(acc_samples.mean()),
        "ci_lower":           float(lower),
        "ci_upper":           float(upper),
        "ci_level":           ci,
        "n_bootstrap":        int(len(eer_samples)),
    }, eer_samples


# ==========================================================================
# 6. DECISION MATRIX & CONFUSION MATRIX
# ==========================================================================

def decision_matrix(score_df: pd.DataFrame, label_df: pd.DataFrame, threshold: float, higher_is_match: bool = True) -> pd.DataFrame:
    scores   = score_df.values
    labels   = label_df.values
    accepted = scores >= threshold if higher_is_match else scores <= threshold
    out      = np.empty(scores.shape, dtype=object)
    out[ labels &  accepted] = "TP"
    out[ labels & ~accepted] = "FN"
    out[~labels &  accepted] = "FP"
    out[~labels & ~accepted] = "TN"
    return pd.DataFrame(out, index=score_df.index, columns=score_df.columns)


def confusion_matrix_counts(decision_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Aggregates decision matrix into 2x2 confusion matrix with Accuracy = (TP+TN)/(TP+TN+FP+FN)."""
    flat = decision_df.values.flatten()
    tp = int(np.sum(flat == "TP"))
    fn = int(np.sum(flat == "FN"))
    fp = int(np.sum(flat == "FP"))
    tn = int(np.sum(flat == "TN"))
    total = max(tp + fn + fp + tn, 1)

    counts_df = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["Actual: Genuine", "Actual: Impostor"],
        columns=["Predicted: Accept", "Predicted: Reject"],
    )
    metrics = {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "accuracy":    (tp + tn) / total,      # (TP + TN) / (TP + TN + FP + FN)
        "precision":   tp / max(tp + fp, 1),
        "recall":      tp / max(tp + fn, 1),   # = GAR
        "specificity": tn / max(tn + fp, 1),   # = 1 - FAR
    }
    return counts_df, metrics


def per_person_metrics(person_ids: List[str], score_df: pd.DataFrame, higher_is_match: bool = True) -> pd.DataFrame:
    rows   = []
    scores = score_df.values

    for idx, pid in enumerate(person_ids):
        row             = scores[idx, :]
        genuine_score   = row[idx]
        impostor_scores = np.delete(row, idx)
        candidate_thresholds = np.linspace(row.min(), row.max(), 500)
        best_t, best_diff, best_far, best_frr, best_acc = None, None, None, None, None

        for t in candidate_thresholds:
            far = np.mean(impostor_scores >= t) if higher_is_match else np.mean(impostor_scores <= t)
            frr = (1.0 if genuine_score < t else 0.0) if higher_is_match else (1.0 if genuine_score > t else 0.0)
            diff = abs(far - frr)
            if best_diff is None or diff < best_diff:
                best_diff, best_t, best_far, best_frr = diff, t, far, frr
                tp = 1 if frr == 0.0 else 0
                tn = int(np.sum(impostor_scores < t)) if higher_is_match else int(np.sum(impostor_scores > t))
                best_acc = (tp + tn) / float(1 + len(impostor_scores))

        rows.append({
            "person_id":           pid,
            "genuine_score":       genuine_score,
            "mean_impostor_score": impostor_scores.mean(),
            "max_impostor_score":  impostor_scores.max() if higher_is_match else impostor_scores.min(),
            "local_threshold":     best_t,
            "local_FAR":           best_far,
            "local_FRR":           best_frr,
            "local_accuracy":      best_acc,
        })

    return pd.DataFrame(rows)


def threshold_journey_matrix(
    person_ids: List[str],
    data: Dict[str, Any],
    eer_threshold: float,
    metric_name: str = "cosine",
    num_thresholds: int = 1000,
) -> pd.DataFrame:
    metric_fn = METRICS[metric_name]["func"]
    higher_is_match = METRICS[metric_name]["higher_is_match"]

    enrollment = {pid: data[pid]["enrollment"] for pid in person_ids}
    live_sets  = {pid: [emb for _, emb in data[pid]["live_sets"]] for pid in person_ids}

    person_gen_means = []
    for pid in person_ids:
        samples = live_sets[pid] if live_sets[pid] else [enrollment[pid]]
        scores = [metric_fn(s, enrollment[pid]) for s in samples]
        person_gen_means.append(float(np.mean(scores)))

    all_scores = np.array(person_gen_means)
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)
    step_size = thresholds[1] - thresholds[0] if len(thresholds) > 1 else 0.001

    rows = []
    for t in thresholds:
        if higher_is_match:
            decisions = ["ACCEPT" if s >= t else "REJECT" for s in person_gen_means]
        else:
            decisions = ["ACCEPT" if s <= t else "REJECT" for s in person_gen_means]

        row = {
            "threshold":  round(float(t), 6),
            "is_EER_thr": "<-- EER" if abs(t - eer_threshold) < step_size else "",
        }
        for pid, dec in zip(person_ids, decisions):
            row[pid] = dec
        row["n_accepted"] = decisions.count("ACCEPT")
        row["n_rejected"] = decisions.count("REJECT")
        rows.append(row)

    return pd.DataFrame(rows)


# ==========================================================================
# 7. PLOTTING FUNCTIONS
# ==========================================================================

def plot_det_curve(sweep_df: pd.DataFrame, out_path_plot: str, out_path_data: str, label: str = ""):
    sweep_df[["threshold", "FAR", "FRR", "accuracy"]].to_csv(out_path_data, index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    far = np.clip(sweep_df["FAR"].values, 1e-4, 1.0)
    frr = np.clip(sweep_df["FRR"].values, 1e-4, 1.0)
    ax.plot(far, frr, color="steelblue", linewidth=1.8, label="Fused DET")
    
    # Mark EER
    eer_idx = sweep_df["abs_diff"].idxmin()
    ax.plot(
        sweep_df.loc[eer_idx, "FAR"], sweep_df.loc[eer_idx, "FRR"],
        "ro", markersize=7, label=f"EER = {sweep_df.loc[eer_idx, 'FAR']*100:.2f}%"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("False Reject Rate (FRR)")
    ax.set_title(f"Fused DET Curve{' - ' + label if label else ''}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


def plot_roc_curve(sweep_df: pd.DataFrame, out_path_plot: str, out_path_data: str, label: str = ""):
    sweep_df[["threshold", "FAR", "TAR", "accuracy"]].to_csv(out_path_data, index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    order = np.argsort(sweep_df["FAR"].values)
    ax.plot(sweep_df["FAR"].values[order], sweep_df["TAR"].values[order], color="darkorange", linewidth=1.8, label="Fused ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    
    eer_idx = sweep_df["abs_diff"].idxmin()
    ax.plot(
        sweep_df.loc[eer_idx, "FAR"], sweep_df.loc[eer_idx, "TAR"],
        "ro", markersize=7, label=f"EER Operating Point (TAR = {sweep_df.loc[eer_idx, 'TAR']*100:.2f}%)"
    )

    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR / GAR)")
    ax.set_title(f"Fused ROC Curve{' - ' + label if label else ''}")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


def plot_score_distribution(genuine_scores: np.ndarray, impostor_scores: np.ndarray, out_path_plot: str, out_path_data: str, metric_label: str = "", threshold: Optional[float] = None):
    max_len = max(len(genuine_scores), len(impostor_scores))
    gen_padded = np.full(max_len, np.nan)
    imp_padded = np.full(max_len, np.nan)
    gen_padded[:len(genuine_scores)] = genuine_scores
    imp_padded[:len(impostor_scores)] = impostor_scores
    pd.DataFrame({"genuine_scores": gen_padded, "impostor_scores": imp_padded}).to_csv(out_path_data, index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(impostor_scores, bins=60, alpha=0.55, label=f"Impostor (n={len(impostor_scores)})", color="firebrick", density=True)
    ax.hist(genuine_scores, bins=60, alpha=0.55, label=f"Genuine (n={len(genuine_scores)})", color="seagreen", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"EER Thr = {threshold:.4f}")
    ax.set_xlabel(metric_label or "Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Fused Genuine vs Impostor Score Distribution{' - ' + metric_label if metric_label else ''}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(counts_df: pd.DataFrame, out_path_plot: str, out_path_data: str, label: str = "", accuracy: Optional[float] = None):
    counts_df.to_csv(out_path_data)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(counts_df.values, cmap="Blues")
    ax.set_xticks(range(len(counts_df.columns)))
    ax.set_xticklabels(counts_df.columns, rotation=15, ha="right")
    ax.set_yticks(range(len(counts_df.index)))
    ax.set_yticklabels(counts_df.index)

    total = counts_df.values.sum()
    for i in range(counts_df.shape[0]):
        for j in range(counts_df.shape[1]):
            val = counts_df.values[i, j]
            pct = (val / total) * 100.0
            ax.text(j, i, f"{val}\n({pct:.2f}%)", ha="center", va="center",
                    color="white" if val > counts_df.values.max() / 2 else "black",
                    fontsize=11, fontweight="bold")
                    
    title = f"Fused Confusion Matrix{' - ' + label if label else ''}"
    if accuracy is not None:
        title += f"\nAccuracy = {accuracy*100:.2f}%  ((TP+TN)/Total)"
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


def plot_decision_matrix(decision_df: pd.DataFrame, out_path_plot: str, out_path_data: str, label: str = ""):
    decision_df.to_csv(out_path_data)

    code_map = {"TP": 0, "FN": 1, "FP": 2, "TN": 3}
    coded = decision_df.replace(code_map).values.astype(float)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"])
    im = ax.imshow(coded, cmap=cmap, vmin=-0.5, vmax=3.5)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
    cbar.ax.set_yticklabels(["TP", "FN", "FP", "TN"])
    ax.set_title(f"Fused Decision Matrix (per person-pair){' - ' + label if label else ''}")
    ax.set_xlabel("Enrolled Template (Person)")
    ax.set_ylabel("Probe (Person)")
    if len(decision_df) <= 40:
        ax.set_xticks(range(len(decision_df.columns)))
        ax.set_xticklabels(decision_df.columns, rotation=90, fontsize=5)
        ax.set_yticks(range(len(decision_df.index)))
        ax.set_yticklabels(decision_df.index, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


def plot_bootstrap_distribution(eer_samples: np.ndarray, ci_info: Dict[str, float], out_path_plot: str, out_path_data: str, label: str = ""):
    pd.DataFrame({"bootstrap_eer": eer_samples}).to_csv(out_path_data, index=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(eer_samples, bins=40, color="slateblue", alpha=0.75)
    ax.axvline(ci_info["eer_bootstrap_mean"], color="black", linestyle="-", label="Bootstrap mean")
    ax.axvline(ci_info["ci_lower"], color="red", linestyle="--", label=f"{ci_info['ci_level']}% CI")
    ax.axvline(ci_info["ci_upper"], color="red", linestyle="--")
    ax.set_xlabel("EER")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Fused Bootstrap EER Distribution{' - ' + label if label else ''}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


# ==========================================================================
# 8. PER-METRIC FUSED EVALUATION
# ==========================================================================

def evaluate_fused_metric(
    person_ids: List[str],
    data: Dict[str, Any],
    metric_name: str,
    out_dir: str,
    n_bootstrap: int = 500,
    num_thresholds: int = 1000,
) -> Dict[str, Any]:
    metric_info = METRICS[metric_name]
    higher_is_match = metric_info["higher_is_match"]
    label = metric_info["label"]

    metric_dir = os.path.join(out_dir, metric_name)
    os.makedirs(metric_dir, exist_ok=True)

    print(f"\n{'-'*60}\nFused Metric: {label}\n{'-'*60}")

    # 1. Scores & Matrices
    gen_scores, imp_scores, gen_by_person, imp_by_person, score_df = \
        compute_fused_scores_and_matrices(person_ids, data, metric_name)

    score_df.to_csv(os.path.join(metric_dir, f"fused_{metric_name}_score_matrix.csv"))
    label_df = genuine_impostor_label_matrix(person_ids)
    label_df.to_csv(os.path.join(metric_dir, f"fused_{metric_name}_genuine_impostor_label_matrix.csv"))

    print(f"  Genuine pairs : {len(gen_scores)}   mean={gen_scores.mean():.4f}")
    print(f"  Impostor pairs: {len(imp_scores)}   mean={imp_scores.mean():.4f}")

    # 2. Threshold sweep + EER + Accuracy
    sweep_df = threshold_sweep(gen_scores, imp_scores, higher_is_match=higher_is_match, num_thresholds=num_thresholds)
    sweep_df.to_csv(os.path.join(metric_dir, f"fused_{metric_name}_threshold_sweep.csv"), index=False)
    eer, eer_thr = compute_eer(sweep_df)

    # 3. TAR @ fixed FAR operating points
    tar_table = tar_at_far_targets(sweep_df, far_targets=(0.01, 0.001, 0.0001), higher_is_match=higher_is_match)
    tar_table.to_csv(os.path.join(metric_dir, f"fused_{metric_name}_tar_at_far.csv"), index=False)

    # 4. Decidability index (d-prime)
    dprime = d_prime(gen_scores, imp_scores)

    # 5. Person-level subject bootstrap CI for EER & Accuracy
    ci_info, eer_samples = bootstrap_eer_person_level(
        person_ids, gen_by_person, imp_by_person,
        higher_is_match=higher_is_match, n_bootstrap=n_bootstrap, ci=95,
        num_thresholds=max(200, num_thresholds // 4),
    )

    # 6. Decision matrix & Confusion matrix at EER threshold
    decision_df = decision_matrix(score_df, label_df, eer_thr, higher_is_match=higher_is_match)
    counts_df, cm_metrics = confusion_matrix_counts(decision_df)

    # 7. Per-person local metrics
    per_person_df = per_person_metrics(person_ids, score_df, higher_is_match=higher_is_match)
    per_person_df.to_csv(os.path.join(metric_dir, f"fused_{metric_name}_per_person_metrics.csv"), index=False)

    # 8. Threshold Journey Matrix
    journey_df = threshold_journey_matrix(
        person_ids, data,
        eer_threshold=eer_thr,
        metric_name=metric_name,
        num_thresholds=num_thresholds,
    )
    journey_df.to_csv(os.path.join(metric_dir, f"fused_{metric_name}_threshold_journey_matrix.csv"), index=False)

    # 9. Visualizations & plots
    plot_det_curve(
        sweep_df,
        os.path.join(metric_dir, f"fused_{metric_name}_det_curve.png"),
        os.path.join(metric_dir, f"fused_{metric_name}_det_curve_data.csv"),
        label=label,
    )
    plot_roc_curve(
        sweep_df,
        os.path.join(metric_dir, f"fused_{metric_name}_roc_curve.png"),
        os.path.join(metric_dir, f"fused_{metric_name}_roc_curve_data.csv"),
        label=label,
    )
    plot_score_distribution(
        gen_scores, imp_scores,
        os.path.join(metric_dir, f"fused_{metric_name}_score_distribution.png"),
        os.path.join(metric_dir, f"fused_{metric_name}_score_distribution_data.csv"),
        metric_label=label,
        threshold=eer_thr,
    )
    plot_confusion_matrix(
        counts_df,
        os.path.join(metric_dir, f"fused_{metric_name}_confusion_matrix.png"),
        os.path.join(metric_dir, f"fused_{metric_name}_confusion_matrix_data.csv"),
        label=label,
        accuracy=cm_metrics["accuracy"],
    )
    plot_decision_matrix(
        decision_df,
        os.path.join(metric_dir, f"fused_{metric_name}_decision_matrix.png"),
        os.path.join(metric_dir, f"fused_{metric_name}_decision_matrix_data.csv"),
        label=label,
    )
    plot_bootstrap_distribution(
        eer_samples, ci_info,
        os.path.join(metric_dir, f"fused_{metric_name}_bootstrap_eer_distribution.png"),
        os.path.join(metric_dir, f"fused_{metric_name}_bootstrap_eer_distribution_data.csv"),
        label=label,
    )

    # Print summary metrics for this metric
    acc_pct = cm_metrics["accuracy"] * 100.0
    print(f"  EER = {eer*100:.3f}%  @ threshold = {eer_thr:.4f}")
    print(f"  Accuracy (TP+TN)/(TP+TN+FP+FN) = {acc_pct:.2f}%  (TP={cm_metrics['TP']}, TN={cm_metrics['TN']}, FP={cm_metrics['FP']}, FN={cm_metrics['FN']})")
    print(f"  d-prime (decidability) = {dprime:.4f}")
    print(f"  Bootstrap EER (95% CI): {ci_info['ci_lower']*100:.3f}% - {ci_info['ci_upper']*100:.3f}% (mean: {ci_info['eer_bootstrap_mean']*100:.3f}%)")

    return {
        "metric":                  metric_name,
        "n_persons":               len(person_ids),
        "genuine_pairs":           len(gen_scores),
        "impostor_pairs":          len(imp_scores),
        "genuine_mean":            round(float(gen_scores.mean()), 6),
        "impostor_mean":           round(float(imp_scores.mean()), 6),
        "EER":                     eer,
        "EER_pct":                 round(eer * 100, 4),
        "EER_threshold":           eer_thr,
        "accuracy_at_EER":         cm_metrics["accuracy"],
        "accuracy_pct":            round(acc_pct, 4),
        "TP":                      cm_metrics["TP"],
        "TN":                      cm_metrics["TN"],
        "FP":                      cm_metrics["FP"],
        "FN":                      cm_metrics["FN"],
        "precision_at_EER":        cm_metrics["precision"],
        "GAR_at_EER":              cm_metrics["recall"],
        "specificity_at_EER":      cm_metrics["specificity"],
        "d_prime":                 dprime,
        "EER_bootstrap_mean":      ci_info["eer_bootstrap_mean"],
        "EER_CI_lower":            ci_info["ci_lower"],
        "EER_CI_upper":            ci_info["ci_upper"],
        "TAR@FAR=1%":              float(tar_table.loc[tar_table["target_FAR"] == 0.01, "TAR"].values[0]),
        "TAR@FAR=0.1%":            float(tar_table.loc[tar_table["target_FAR"] == 0.001, "TAR"].values[0]),
        "TAR@FAR=0.01%":           float(tar_table.loc[tar_table["target_FAR"] == 0.0001, "TAR"].values[0]),
    }


# ==========================================================================
# 9. MAIN ENTRY POINT
# ==========================================================================

def main(
    db_path: str,
    dataset: str,
    fusion_type: str,
    out_dir: str,
    template_type: str = "raw",
    metrics: Tuple[str, ...] = ("cosine", "euclidean", "manhattan", "hamming"),
    n_bootstrap: int = 500,
    num_thresholds: int = 1000,
):
    os.makedirs(out_dir, exist_ok=True)

    data = load_fused_data(db_path, dataset, fusion_type, template_type=template_type)
    person_ids = sorted(data.keys())

    sep = "=" * 65
    print(f"\n{sep}\nEVALUATING FUSED BIOMETRIC EMBEDDINGS ({fusion_type.upper()}, type={template_type.upper()})\n{sep}")
    print(f"  Dataset         : {dataset}")
    print(f"  Persons         : {len(person_ids)}")
    print(f"  Embedding Vector: {data[person_ids[0]]['enrollment'].shape[0]}-D")
    print(f"  Metrics to Run  : {', '.join(metrics)}")

    summary_rows = []
    for metric_name in metrics:
        if metric_name not in METRICS:
            print(f"[WARN] Unknown metric '{metric_name}', skipping.")
            continue
        res = evaluate_fused_metric(
            person_ids, data, metric_name, out_dir,
            n_bootstrap=n_bootstrap, num_thresholds=num_thresholds
        )
        res["fusion_type"] = fusion_type
        res["template_type"] = template_type
        summary_rows.append(res)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "fused_evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{sep}\nFUSED BIOMETRIC BENCHMARK RESULTS SUMMARY\n{sep}")
    print(summary_df[[
        "metric", "EER_pct", "accuracy_pct", "EER_threshold", "d_prime", "TAR@FAR=1%", "TAR@FAR=0.1%"
    ]].to_string(index=False))

    print(f"\n[DONE] All fused evaluation plots, matrices, and summary tables saved under:\n       {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive Fused Biometric Embeddings Evaluation Suite."
    )
    parser.add_argument("--db",            type=str, default="database/biometric.db",
                        help="Path to SQLite database.")
    parser.add_argument("--dataset",       type=str, default="setB",
                        help="Dataset tag (e.g. setA, setB).")
    parser.add_argument("--fusion_type",   type=str, default="concat_l2",
                        help="Fusion type stored in database (e.g. concat_l2, cbp).")
    parser.add_argument("--template_type", type=str, default="raw", choices=["raw", "biohash"],
                        help="Type of template to evaluate: 'raw' or 'biohash'.")
    parser.add_argument("--out_dir",       type=str, default="results/fused_eval",
                        help="Output directory for fused evaluation matrices.")
    parser.add_argument("--metrics",       type=str, nargs="+",
                        default=["cosine","hamming"],
                        help="Which metrics to evaluate. choice left :euclidean, manhattan")
    parser.add_argument("--n_bootstrap",   type=int, default=500,
                        help="Number of person-level bootstrap iterations for EER & Accuracy CI.")
    parser.add_argument("--num_thresholds", type=int, default=1000)
    args = parser.parse_args()

    main(
        args.db, args.dataset, args.fusion_type, args.out_dir,
        template_type=args.template_type,
        metrics=tuple(args.metrics),
        n_bootstrap=args.n_bootstrap,
        num_thresholds=args.num_thresholds,
    )
