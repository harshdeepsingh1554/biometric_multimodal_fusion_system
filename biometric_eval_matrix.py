"""
biometric_full_eval.py — Comprehensive Per-Modality Biometric Evaluation Suite
=================================================================================

Evaluates raw and BioHashed biometric templates PER TRAIT/MODALITY (no fusion).

Data source (matches biohashing.py's actual schema):
    Raw embeddings:
        - EMBEDDINGS          (person_id, dataset, trait, embedding, dim)      -> enrollment templates
        - live_embeddings     (person_id, dataset, live_index, trait, embedding, dim) -> probes
    BioHashed embeddings:
        - biohash_embeddings       (person_id, dataset, trait, hash_dim, embedding, dim) -> enrollment templates
        - biohash_live_embeddings  (person_id, dataset, live_index, trait, hash_dim, embedding, dim) -> probes

    Similarity / distance measures:
        - Cosine similarity
        - Euclidean distance
        - Manhattan (L1) distance
        - Hamming distance (sign-binarized; exact for BioHash bipolar codes)

    Accuracy metrics (per metric, per trait):
        - Genuine / impostor score arrays
        - Threshold sweep: FAR, FRR, GAR (=1-FRR), TAR (=GAR)
        - EER + EER threshold
        - Person-level bootstrap confidence interval on EER
        - TAR @ fixed FAR operating points (1%, 0.1%, 0.01%)
        - d-prime (decidability index)
        - Confusion matrix (TP/FP/TN/FN counts at EER threshold)
        - Full pairwise decision matrix (TP/FN/FP/TN per person pair)

    Graphs (each saved WITH its underlying data table):
        - DET curve (FAR vs FRR, log-log)
        - ROC curve (FAR vs TAR)
        - Genuine vs impostor score distribution histogram
        - Confusion matrix heatmap
        - Decision matrix heatmap (per-person-pair TP/FN/FP/TN)

Usage:
    python biometric_full_eval.py --db database/biometric.db --dataset setA \
        --template_type biohash --out_dir results/full_eval

    # Evaluate only one trait (e.g. "face") instead of every trait present:
    python biometric_full_eval.py --db database/biometric.db --dataset setA \
        --template_type raw --trait face --out_dir results/full_eval

Notes on the bootstrap method
------------------------------
A NAIVE bootstrap that resamples individual genuine/impostor SCORES treats
every score as independent. In reality scores are clustered by person (one
person contributes multiple genuine scores and multiple impostor scores),
so a flat score-level bootstrap understates the true uncertainty.

This module instead performs a PERSON-LEVEL (subject) bootstrap: on each
iteration it resamples the person IDs WITH REPLACEMENT, and reassembles
the genuine/impostor score pools from only the resampled persons' own
contributions. This preserves the person-level clustering structure.

Simplification (documented, not hidden): if a person is drawn more than
once in a resampling iteration, this implementation does NOT synthesize new
"genuine" comparisons between the duplicate copies of that person (which a
full identity-recombination bootstrap would do). It only re-pools each
drawn person's ALREADY-COMPUTED genuine and impostor scores. This is a
standard, conservative approximation used when re-deriving full pairwise
comparisons for resampled identities is impractical.
"""

import argparse
import os
import sqlite3
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================================
# 1. LOAD DATA FROM SQLITE (per trait/modality, no fusion)
# ==========================================================================

def load_data(db_path: str, dataset: str, template_type: str = "raw", trait_filter: Optional[str] = None):
    """
    Loads enrollment templates + live/probe embeddings, grouped by trait.

    template_type="raw"     -> reads EMBEDDINGS / live_embeddings
    template_type="biohash" -> reads biohash_embeddings / biohash_live_embeddings

    Returns
    -------
    dict: trait -> { person_id -> {"enrollment": np.ndarray, "live_sets": [(idx, np.ndarray), ...]} }
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    if template_type == "biohash":
        tmpl_table = "biohash_embeddings"
        live_table = "biohash_live_embeddings"
    elif template_type == "raw":
        tmpl_table = "EMBEDDINGS"
        live_table = "live_embeddings"
    else:
        raise ValueError(f"Unknown template_type '{template_type}'. Use 'raw' or 'biohash'.")

    if tmpl_table not in tables:
        conn.close()
        raise ValueError(
            f"Table '{tmpl_table}' not found in {db_path}. Available tables: {tables}"
        )

    tmpl_query = f"SELECT person_id, trait, embedding, dim FROM {tmpl_table} WHERE dataset=?"
    params = [dataset]
    if trait_filter:
        tmpl_query += " AND trait=?"
        params.append(trait_filter)
    tmpl_query += " ORDER BY trait, person_id"

    tmpl_rows = cur.execute(tmpl_query, tuple(params)).fetchall()

    live_rows = []
    if live_table in tables:
        live_query = f"SELECT person_id, live_index, trait, embedding, dim FROM {live_table} WHERE dataset=?"
        live_params = [dataset]
        if trait_filter:
            live_query += " AND trait=?"
            live_params.append(trait_filter)
        live_query += " ORDER BY trait, person_id, live_index"
        live_rows = cur.execute(live_query, tuple(live_params)).fetchall()
    else:
        print(f"[WARN] Table '{live_table}' not found in {db_path}. "
              f"No probe/live embeddings loaded for template_type='{template_type}' "
              f"(enrollment vectors will be used as their own probe, which inflates genuine scores).")

    conn.close()

    if not tmpl_rows:
        scope = f", trait='{trait_filter}'" if trait_filter else ""
        raise ValueError(
            f"No templates found in '{tmpl_table}' for dataset='{dataset}'{scope}, "
            f"template_type='{template_type}'."
        )

    data = {}  # trait -> {person_id: {"enrollment": vec, "live_sets": [(idx, vec), ...]}}
    for pid, trait, blob, dim in tmpl_rows:
        emb = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
        data.setdefault(trait, {}).setdefault(pid, {"enrollment": None, "live_sets": []})
        data[trait][pid]["enrollment"] = emb

    for pid, l_idx, trait, blob, dim in live_rows:
        if trait in data and pid in data[trait]:
            emb = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
            data[trait][pid]["live_sets"].append((l_idx, emb))

    for trait, persons in data.items():
        n_live = sum(len(v["live_sets"]) for v in persons.values())
        missing_live = sum(1 for v in persons.values() if not v["live_sets"])
        print(f"[DB] Trait '{trait}': {len(persons)} persons loaded from '{tmpl_table}', "
              f"{n_live} live/probe embeddings from '{live_table}'  (type={template_type.upper()})")
        if missing_live:
            print(f"     [WARN] {missing_live} person(s) in trait '{trait}' have NO live/probe samples "
                  f"-> their enrollment vector will be used as its own probe.")

    return data


# ==========================================================================
# 2. SIMILARITY / DISTANCE METRICS
# ==========================================================================
# Every metric function takes two 1-D numpy arrays and returns a scalar.
# "higher_is_match" tells the rest of the pipeline whether a HIGHER score
# means a better match (similarity) or a LOWER score means a better match
# (distance).

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_n, b_n))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalized Hamming distance in [0, 1] (fraction of differing bits).

    For already-bipolar/binary vectors (e.g. BioHash {-1,+1} codes) this
    operates directly on the sign of each element, which is exact for that
    encoding. For raw continuous embeddings this sign-binarizes first as a
    reasonable default, but Hamming distance is primarily meaningful for
    hashed/binary templates, not raw embeddings -- interpret with that in
    mind when applying it to 'raw' template_type data.
    """
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
# 3. SCORE MATRIX + GENUINE/IMPOSTOR EXTRACTION (per metric, per trait)
# ==========================================================================

def compute_scores_and_matrices(person_ids, trait_data, metric_name):
    """
    For the given metric, computes (within a single trait's data):
      - full person x person score matrix (mean over live samples)
      - flat genuine score array (with per-person grouping preserved for bootstrap)
      - flat impostor score array (with per-probe-person grouping preserved for bootstrap)
    """
    metric_fn = METRICS[metric_name]["func"]

    enrollment = {pid: trait_data[pid]["enrollment"] for pid in person_ids}
    live_sets  = {pid: [emb for _, emb in trait_data[pid]["live_sets"]] for pid in person_ids}

    n = len(person_ids)
    score_mat = np.zeros((n, n))

    gen_scores, imp_scores = [], []
    gen_by_person = {pid: [] for pid in person_ids}   # for person-level bootstrap
    imp_by_person = {pid: [] for pid in person_ids}   # keyed by the PROBE person (row)

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


def genuine_impostor_label_matrix(person_ids):
    n = len(person_ids)
    labels = np.eye(n, dtype=bool)
    return pd.DataFrame(labels, index=person_ids, columns=person_ids)


# ==========================================================================
# 4. THRESHOLD SWEEP, EER, GAR/TAR, D-PRIME
# ==========================================================================

def threshold_sweep(genuine_scores, impostor_scores, higher_is_match=True, num_thresholds=1000):
    """FAR/FRR sweep, with GAR and TAR added (GAR = TAR = 1 - FRR)."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)
    n_gen, n_imp = len(genuine_scores), len(impostor_scores)
    far_list, frr_list = [], []

    for t in thresholds:
        if higher_is_match:
            far = np.sum(impostor_scores >= t) / n_imp if n_imp else 0.0
            frr = np.sum(genuine_scores  <  t) / n_gen if n_gen else 0.0
        else:
            far = np.sum(impostor_scores <= t) / n_imp if n_imp else 0.0
            frr = np.sum(genuine_scores  >  t) / n_gen if n_gen else 0.0
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    return pd.DataFrame({
        "threshold": thresholds,
        "FAR":       far_arr,
        "FRR":       frr_arr,
        "GAR":       1.0 - frr_arr,   # Genuine Accept Rate
        "TAR":       1.0 - frr_arr,   # True Accept Rate (identical definition to GAR)
        "abs_diff":  np.abs(far_arr - frr_arr),
    })


def compute_eer(sweep_df):
    idx = sweep_df["abs_diff"].idxmin()
    row = sweep_df.loc[idx]
    return float((row["FAR"] + row["FRR"]) / 2.0), float(row["threshold"])


def tar_at_far_targets(sweep_df, far_targets=(0.01, 0.001, 0.0001), higher_is_match=True):
    """
    For each target FAR (e.g. 1%, 0.1%, 0.01%), find the operating point in the
    sweep whose FAR is closest to (and at or below, where possible) the target,
    and report the TAR/threshold there. This is the standard way biometric
    systems are compared at fixed security operating points, rather than only
    at the single EER crossover point.
    """
    rows = []
    df_sorted = sweep_df.sort_values("threshold", ascending=not higher_is_match).reset_index(drop=True)

    for target in far_targets:
        candidates = df_sorted[df_sorted["FAR"] <= target]
        if len(candidates) > 0:
            # Tightest (largest) FAR that still satisfies <= target -> least conservative valid point
            row = candidates.iloc[candidates["FAR"].values.argmax()]
            achieved = True
        else:
            # No threshold achieves this FAR; report the closest available point instead
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


def d_prime(genuine_scores, impostor_scores):
    """
    Decidability index: separation between genuine and impostor distributions
    in pooled standard-deviation units. Higher = better separated / more
    discriminative system. Threshold-independent, unlike EER.
    """
    mu_g, mu_i = genuine_scores.mean(), impostor_scores.mean()
    var_g, var_i = genuine_scores.var(), impostor_scores.var()
    denom = np.sqrt((var_g + var_i) / 2.0)
    if denom < 1e-12:
        return 0.0
    return float(abs(mu_g - mu_i) / denom)


# ==========================================================================
# 5. PERSON-LEVEL BOOTSTRAP CONFIDENCE INTERVAL FOR EER
# ==========================================================================

def bootstrap_eer_person_level(
    person_ids, gen_by_person, imp_by_person,
    higher_is_match=True, n_bootstrap=1000, ci=95, num_thresholds=500, seed=42
):
    """
    Person-level (subject) bootstrap for EER confidence intervals.
    See module docstring for the method and its documented simplification.
    """
    rng = np.random.RandomState(seed)
    eer_samples = []
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
        eer, _ = compute_eer(sweep_df)
        eer_samples.append(eer)

    eer_samples = np.array(eer_samples)
    lower = np.percentile(eer_samples, (100 - ci) / 2)
    upper = np.percentile(eer_samples, 100 - (100 - ci) / 2)

    return {
        "eer_bootstrap_mean": float(eer_samples.mean()),
        "eer_bootstrap_std":  float(eer_samples.std()),
        "ci_lower":           float(lower),
        "ci_upper":           float(upper),
        "ci_level":           ci,
        "n_bootstrap":        int(len(eer_samples)),
    }, eer_samples


# ==========================================================================
# 6. DECISION MATRIX & CONFUSION MATRIX
# ==========================================================================

def decision_matrix(score_df, label_df, threshold, higher_is_match=True):
    scores   = score_df.values
    labels   = label_df.values
    accepted = scores >= threshold if higher_is_match else scores <= threshold
    out      = np.empty(scores.shape, dtype=object)
    out[ labels &  accepted] = "TP"
    out[ labels & ~accepted] = "FN"
    out[~labels &  accepted] = "FP"
    out[~labels & ~accepted] = "TN"
    return pd.DataFrame(out, index=score_df.index, columns=score_df.columns)


def confusion_matrix_counts(decision_df):
    """Aggregates the per-pair decision matrix into a single 2x2 confusion count table."""
    flat = decision_df.values.flatten()
    tp = int(np.sum(flat == "TP"))
    fn = int(np.sum(flat == "FN"))
    fp = int(np.sum(flat == "FP"))
    tn = int(np.sum(flat == "TN"))

    counts_df = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["Actual: Genuine", "Actual: Impostor"],
        columns=["Predicted: Accept", "Predicted: Reject"],
    )
    metrics = {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "accuracy":    (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision":   tp / max(tp + fp, 1),
        "recall":      tp / max(tp + fn, 1),   # = GAR
        "specificity": tn / max(tn + fp, 1),   # = 1 - FAR
    }
    return counts_df, metrics


# ==========================================================================
# 7. PLOTTING (each plot saved alongside its underlying data table)
# ==========================================================================

def plot_det_curve(sweep_df, out_path_plot, out_path_data, label=""):
    sweep_df[["threshold", "FAR", "FRR"]].to_csv(out_path_data, index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    far = np.clip(sweep_df["FAR"].values, 1e-4, 1.0)
    frr = np.clip(sweep_df["FRR"].values, 1e-4, 1.0)
    ax.plot(far, frr, color="steelblue", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("False Reject Rate (FRR)")
    ax.set_title(f"DET Curve{' - ' + label if label else ''}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


def plot_roc_curve(sweep_df, out_path_plot, out_path_data, label=""):
    sweep_df[["threshold", "FAR", "TAR"]].to_csv(out_path_data, index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    order = np.argsort(sweep_df["FAR"].values)
    ax.plot(sweep_df["FAR"].values[order], sweep_df["TAR"].values[order], color="darkorange", linewidth=1.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR)")
    ax.set_title(f"ROC Curve{' - ' + label if label else ''}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


def plot_score_distribution(genuine_scores, impostor_scores, out_path_plot, out_path_data, metric_label=""):
    max_len = max(len(genuine_scores), len(impostor_scores))
    gen_padded = np.full(max_len, np.nan)
    imp_padded = np.full(max_len, np.nan)
    gen_padded[:len(genuine_scores)] = genuine_scores
    imp_padded[:len(impostor_scores)] = impostor_scores
    pd.DataFrame({"genuine_scores": gen_padded, "impostor_scores": imp_padded}).to_csv(out_path_data, index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(impostor_scores, bins=60, alpha=0.55, label=f"Impostor (n={len(impostor_scores)})", color="firebrick", density=True)
    ax.hist(genuine_scores, bins=60, alpha=0.55, label=f"Genuine (n={len(genuine_scores)})", color="seagreen", density=True)
    ax.set_xlabel(metric_label or "Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Genuine vs Impostor Score Distribution{' - ' + metric_label if metric_label else ''}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(counts_df, out_path_plot, out_path_data, label=""):
    counts_df.to_csv(out_path_data)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(counts_df.values, cmap="Blues")
    ax.set_xticks(range(len(counts_df.columns)))
    ax.set_xticklabels(counts_df.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(counts_df.index)))
    ax.set_yticklabels(counts_df.index)
    for i in range(counts_df.shape[0]):
        for j in range(counts_df.shape[1]):
            ax.text(j, i, str(counts_df.values[i, j]), ha="center", va="center",
                    color="white" if counts_df.values[i, j] > counts_df.values.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    ax.set_title(f"Confusion Matrix (at EER threshold){' - ' + label if label else ''}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


def plot_decision_matrix(decision_df, out_path_plot, out_path_data, label=""):
    decision_df.to_csv(out_path_data)

    code_map = {"TP": 0, "FN": 1, "FP": 2, "TN": 3}
    coded = decision_df.replace(code_map).values.astype(float)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"])
    im = ax.imshow(coded, cmap=cmap, vmin=-0.5, vmax=3.5)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
    cbar.ax.set_yticklabels(["TP", "FN", "FP", "TN"])
    ax.set_title(f"Decision Matrix (per person-pair){' - ' + label if label else ''}")
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


def plot_bootstrap_distribution(eer_samples, ci_info, out_path_plot, out_path_data, label=""):
    pd.DataFrame({"bootstrap_eer": eer_samples}).to_csv(out_path_data, index=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(eer_samples, bins=40, color="slateblue", alpha=0.75)
    ax.axvline(ci_info["eer_bootstrap_mean"], color="black", linestyle="-", label="Bootstrap mean")
    ax.axvline(ci_info["ci_lower"], color="red", linestyle="--", label=f"{ci_info['ci_level']}% CI")
    ax.axvline(ci_info["ci_upper"], color="red", linestyle="--")
    ax.set_xlabel("EER")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bootstrap EER Distribution{' - ' + label if label else ''}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path_plot, dpi=150)
    plt.close(fig)


# ==========================================================================
# 8. PER-METRIC EVALUATION PIPELINE (runs within a single trait)
# ==========================================================================

def evaluate_metric(person_ids, trait_data, metric_name, out_dir, n_bootstrap=500, num_thresholds=1000):
    metric_info = METRICS[metric_name]
    higher_is_match = metric_info["higher_is_match"]
    label = metric_info["label"]

    metric_dir = os.path.join(out_dir, metric_name)
    os.makedirs(metric_dir, exist_ok=True)

    print(f"\n{'-'*60}\nMetric: {label}\n{'-'*60}")

    # --- scores & matrices ---
    gen_scores, imp_scores, gen_by_person, imp_by_person, score_df = \
        compute_scores_and_matrices(person_ids, trait_data, metric_name)

    score_df.to_csv(os.path.join(metric_dir, f"{metric_name}_score_matrix.csv"))
    label_df = genuine_impostor_label_matrix(person_ids)
    label_df.to_csv(os.path.join(metric_dir, f"{metric_name}_genuine_impostor_label_matrix.csv"))

    print(f"  Genuine pairs : {len(gen_scores)}   mean={gen_scores.mean():.4f}")
    print(f"  Impostor pairs: {len(imp_scores)}   mean={imp_scores.mean():.4f}")

    # --- threshold sweep + EER ---
    sweep_df = threshold_sweep(gen_scores, imp_scores, higher_is_match=higher_is_match, num_thresholds=num_thresholds)
    sweep_df.to_csv(os.path.join(metric_dir, f"{metric_name}_threshold_sweep.csv"), index=False)
    eer, eer_thr = compute_eer(sweep_df)
    print(f"  EER = {eer*100:.3f}%  @ threshold = {eer_thr:.4f}")

    # --- TAR @ FAR operating points ---
    tar_table = tar_at_far_targets(sweep_df, far_targets=(0.01, 0.001, 0.0001), higher_is_match=higher_is_match)
    tar_table.to_csv(os.path.join(metric_dir, f"{metric_name}_tar_at_far.csv"), index=False)
    print("  TAR @ fixed FAR operating points:")
    print(tar_table.to_string(index=False))

    # --- d-prime ---
    dprime = d_prime(gen_scores, imp_scores)
    print(f"  d-prime (decidability) = {dprime:.4f}")

    # --- person-level bootstrap CI for EER ---
    ci_info, eer_samples = bootstrap_eer_person_level(
        person_ids, gen_by_person, imp_by_person,
        higher_is_match=higher_is_match, n_bootstrap=n_bootstrap, ci=95,
        num_thresholds=max(200, num_thresholds // 4),  # lighter sweep per bootstrap iter for speed
    )
    print(f"  Bootstrap EER = {ci_info['eer_bootstrap_mean']*100:.3f}% "
          f"(95% CI: {ci_info['ci_lower']*100:.3f}%-{ci_info['ci_upper']*100:.3f}%, "
          f"n={ci_info['n_bootstrap']})")

    # --- decision matrix + confusion matrix (at EER threshold) ---
    decision_df = decision_matrix(score_df, label_df, eer_thr, higher_is_match=higher_is_match)
    counts_df, cm_metrics = confusion_matrix_counts(decision_df)

    # --- GRAPHS + underlying data, all saved together ---
    plot_det_curve(
        sweep_df,
        os.path.join(metric_dir, f"{metric_name}_det_curve.png"),
        os.path.join(metric_dir, f"{metric_name}_det_curve_data.csv"),
        label=label,
    )
    plot_roc_curve(
        sweep_df,
        os.path.join(metric_dir, f"{metric_name}_roc_curve.png"),
        os.path.join(metric_dir, f"{metric_name}_roc_curve_data.csv"),
        label=label,
    )
    plot_score_distribution(
        gen_scores, imp_scores,
        os.path.join(metric_dir, f"{metric_name}_score_distribution.png"),
        os.path.join(metric_dir, f"{metric_name}_score_distribution_data.csv"),
        metric_label=label,
    )
    plot_confusion_matrix(
        counts_df,
        os.path.join(metric_dir, f"{metric_name}_confusion_matrix.png"),
        os.path.join(metric_dir, f"{metric_name}_confusion_matrix_data.csv"),
        label=label,
    )
    plot_decision_matrix(
        decision_df,
        os.path.join(metric_dir, f"{metric_name}_decision_matrix.png"),
        os.path.join(metric_dir, f"{metric_name}_decision_matrix_data.csv"),
        label=label,
    )
    plot_bootstrap_distribution(
        eer_samples, ci_info,
        os.path.join(metric_dir, f"{metric_name}_bootstrap_eer_distribution.png"),
        os.path.join(metric_dir, f"{metric_name}_bootstrap_eer_distribution_data.csv"),
        label=label,
    )

    print(f"  Confusion matrix: accuracy={cm_metrics['accuracy']:.4f}  "
          f"precision={cm_metrics['precision']:.4f}  recall(GAR)={cm_metrics['recall']:.4f}  "
          f"specificity(1-FAR)={cm_metrics['specificity']:.4f}")
    print(f"  All outputs saved under: {metric_dir}")

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
        "EER_bootstrap_mean":      ci_info["eer_bootstrap_mean"],
        "EER_CI_lower":            ci_info["ci_lower"],
        "EER_CI_upper":            ci_info["ci_upper"],
        "d_prime":                 dprime,
        "TAR@FAR=1%":              float(tar_table.loc[tar_table["target_FAR"] == 0.01, "TAR"].values[0]),
        "TAR@FAR=0.1%":            float(tar_table.loc[tar_table["target_FAR"] == 0.001, "TAR"].values[0]),
        "TAR@FAR=0.01%":           float(tar_table.loc[tar_table["target_FAR"] == 0.0001, "TAR"].values[0]),
        "accuracy_at_EER":         cm_metrics["accuracy"],
        "precision_at_EER":        cm_metrics["precision"],
        "GAR_at_EER":              cm_metrics["recall"],
        "specificity_at_EER":      cm_metrics["specificity"],
    }


# ==========================================================================
# 9. PER-TRAIT DRIVER (loops metrics for one trait, mirrors old per-fusion loop)
# ==========================================================================

def evaluate_trait(trait, person_ids, trait_data, out_dir, template_type,
                    metrics=("cosine", "euclidean", "manhattan", "hamming"),
                    n_bootstrap=500, num_thresholds=1000):
    trait_out_dir = os.path.join(out_dir, trait)
    os.makedirs(trait_out_dir, exist_ok=True)

    sep = "=" * 60
    print(f"\n{sep}\nTRAIT: {trait.upper()}  (type={template_type.upper()})\n{sep}")
    print(f"  Persons: {len(person_ids)}")
    print(f"  Metrics: {', '.join(metrics)}")

    summary_rows = []
    for metric_name in metrics:
        if metric_name not in METRICS:
            print(f"[WARN] Unknown metric '{metric_name}', skipping.")
            continue
        result = evaluate_metric(person_ids, trait_data, metric_name, trait_out_dir,
                                  n_bootstrap=n_bootstrap, num_thresholds=num_thresholds)
        result["trait"] = trait
        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(trait_out_dir, f"{trait}_evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  Summary for trait '{trait}':")
    print(summary_df.to_string(index=False))
    print(f"  Saved to: {summary_path}")

    return summary_df


# ==========================================================================
# 10. MAIN ENTRY POINT
# ==========================================================================

def main(db_path: str, dataset: str, out_dir: str, template_type: str = "raw", trait: Optional[str] = None,
         metrics=("cosine", "euclidean", "manhattan", "hamming"),
         n_bootstrap=500, num_thresholds=1000):
    os.makedirs(out_dir, exist_ok=True)

    data = load_data(db_path, dataset, template_type=template_type, trait_filter=trait)
    traits = sorted(data.keys())

    sep = "=" * 60
    print(f"\n{sep}\nFULL BIOMETRIC EVALUATION (per-trait, type={template_type.upper()})\n{sep}")
    print(f"  Dataset: {dataset}")
    print(f"  Traits found: {', '.join(traits)}")

    all_summaries = []
    for tr in traits:
        person_ids = sorted(data[tr].keys())
        trait_summary = evaluate_trait(
            tr, person_ids, data[tr], out_dir, template_type,
            metrics=metrics, n_bootstrap=n_bootstrap, num_thresholds=num_thresholds
        )
        all_summaries.append(trait_summary)

    combined_df = pd.concat(all_summaries, ignore_index=True)
    combined_path = os.path.join(out_dir, "full_evaluation_summary.csv")
    combined_df.to_csv(combined_path, index=False)

    print(f"\n{sep}\nSUMMARY ACROSS ALL TRAITS AND METRICS\n{sep}")
    print(combined_df.to_string(index=False))
    print(f"\nFull combined summary saved to: {combined_path}")
    print(f"All per-trait/per-metric graphs and matrices saved under: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-modality biometric evaluation (no fusion): cosine/euclidean/manhattan/hamming, "
                    "GAR/TAR, d-prime, bootstrap EER CI, DET/ROC/distribution/confusion/decision plots, "
                    "for raw or BioHashed templates, one trait at a time."
    )
    parser.add_argument("--db",            type=str, default="database/biometric.db")
    parser.add_argument("--dataset",       type=str, default="setA")
    parser.add_argument("--template_type", type=str, default="biohash", choices=["raw", "biohash"])
    parser.add_argument("--trait",         type=str, default=None,
                        help="Restrict evaluation to a single trait (e.g. 'face'). "
                             "Default: evaluate every trait found in the templates table.")
    parser.add_argument("--out_dir",       type=str, default="results/full_eval")
    parser.add_argument("--metrics",       type=str, nargs="+",
                        default=["cosine", "euclidean", "manhattan", "hamming"],
                        help="Which metrics to evaluate.")
    parser.add_argument("--n_bootstrap",   type=int, default=500,
                        help="Number of person-level bootstrap iterations for EER CI.")
    parser.add_argument("--num_thresholds", type=int, default=1000)
    args = parser.parse_args()

    main(args.db, args.dataset, args.out_dir, args.template_type, args.trait,
         metrics=tuple(args.metrics), n_bootstrap=args.n_bootstrap, num_thresholds=args.num_thresholds)