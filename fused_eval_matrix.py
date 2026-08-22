"""
fused_eval_matrix.py — Fused Embeddings Evaluation & Journey Matrix Generator
=============================================================================

Loads concatenated 1536-dimensional fused embeddings from SQLite DB
(`fused_templates` and `fused_image_embeddings` tables) and performs
complete biometric evaluation:
    1. Person x Person cosine similarity matrix (80 x 80)
    2. Person x Person euclidean distance matrix (80 x 80)
    3. Genuine / Impostor score partitioning
    4. Decision matrix (TP / FN / TN / FP) at EER threshold
    5. Threshold sweep table (threshold, FAR, FRR) for cosine & euclidean
    6. Per-person local threshold, FAR, FRR, local EER
    7. Threshold Journey Matrix (1000 threshold steps x 80 persons ACCEPT/REJECT)
    8. Global EER + EER threshold recommendation

Usage:
    python fused_eval_matrix.py --db database/biometric.db --dataset setA --out_dir results/fused_eval
"""

import argparse
import os
import sqlite3
import numpy as np
import pandas as pd


# ==========================================================================
# 1. LOAD FUSED DATA FROM SQLITE
# ==========================================================================

def load_fused_data(db_path: str, dataset: str, fusion_type: str = "cbp", template_type: str = "raw"):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    if template_type == "biohash":
        tmpl_table = "biohash_embedding_fused" if "biohash_embedding_fused" in tables else "biohash_fused"
        live_table = "biohashed_livefused" if "biohashed_livefused" in tables else "biohash_live_fused"
    else:
        tmpl_table = "fused_templates"
        live_table = "fused_live_embeddings" if "fused_live_embeddings" in tables else "fused_image_embeddings"

    tmpl_rows = cur.execute(
        f"SELECT person_id, embedding, dim FROM {tmpl_table} WHERE dataset=? AND fusion_type=? ORDER BY person_id",
        (dataset, fusion_type)
    ).fetchall()

    live_rows = []
    if live_table in tables:
        cols = [c[1] for c in cur.execute(f"PRAGMA table_info({live_table})").fetchall()]
        idx_col = "live_index" if "live_index" in cols else ("probe_index" if "probe_index" in cols else "image_index")
        live_rows = cur.execute(
            f"SELECT person_id, {idx_col}, embedding, dim FROM {live_table} WHERE dataset=? AND fusion_type=? ORDER BY person_id, {idx_col}",
            (dataset, fusion_type)
        ).fetchall()

    conn.close()

    if not tmpl_rows:
        raise ValueError(
            f"No fused templates found in '{tmpl_table}' for dataset='{dataset}', fusion_type='{fusion_type}', template_type='{template_type}'. "
            f"Run biohashing.py or fusion scripts first!"
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


def compute_fused_scores_and_matrices(person_ids, data):
    """
    Computes genuine/impostor scores across all live set samples and returns:
    - gen_cos, imp_cos, gen_euc, imp_euc arrays
    - cos_df, euc_df mean matrices per person pair
    """
    enrollment = {pid: data[pid]["enrollment"] for pid in person_ids}
    live_sets  = {pid: [emb for _, emb in data[pid]["live_sets"]] for pid in person_ids}

    n = len(person_ids)
    cos_mat = np.zeros((n, n))
    euc_mat = np.zeros((n, n))

    gen_cos, imp_cos = [], []
    gen_euc, imp_euc = [], []

    e_norm_dict = {pid: vec / (np.linalg.norm(vec) + 1e-12) for pid, vec in enrollment.items()}

    for i, pi in enumerate(person_ids):
        samples = live_sets[pi]
        if not samples:
            samples = [enrollment[pi]]

        norm_samples = [s / (np.linalg.norm(s) + 1e-12) for s in samples]

        for j, pj in enumerate(person_ids):
            e_vec = enrollment[pj]
            e_norm = e_norm_dict[pj]

            cos_scores = [float(np.dot(s_norm, e_norm)) for s_norm in norm_samples]
            euc_scores = [float(np.linalg.norm(s - e_vec)) for s in samples]

            cos_mat[i, j] = float(np.mean(cos_scores))
            euc_mat[i, j] = float(np.mean(euc_scores))

            if i == j:
                gen_cos.extend(cos_scores)
                gen_euc.extend(euc_scores)
            else:
                imp_cos.extend(cos_scores)
                imp_euc.extend(euc_scores)

    cos_df = pd.DataFrame(cos_mat, index=person_ids, columns=person_ids)
    euc_df = pd.DataFrame(euc_mat, index=person_ids, columns=person_ids)

    return (
        np.array(gen_cos), np.array(imp_cos),
        np.array(gen_euc), np.array(imp_euc),
        cos_df, euc_df
    )


# ==========================================================================
# 2. SIMILARITY / DISTANCE MATRICES & LABELS
# ==========================================================================

def genuine_impostor_label_matrix(person_ids):
    n      = len(person_ids)
    labels = np.eye(n, dtype=bool)
    return pd.DataFrame(labels, index=person_ids, columns=person_ids)


# ==========================================================================
# 3. THRESHOLD SWEEP + EER
# ==========================================================================

def threshold_sweep(genuine_scores, impostor_scores, higher_is_match=True, num_thresholds=1000):
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

    return pd.DataFrame({
        "threshold": thresholds,
        "FAR":       far_list,
        "FRR":       frr_list,
        "abs_diff":  np.abs(np.array(far_list) - np.array(frr_list)),
    })


def compute_eer(sweep_df):
    idx = sweep_df["abs_diff"].idxmin()
    row = sweep_df.loc[idx]
    return float((row["FAR"] + row["FRR"]) / 2.0), float(row["threshold"])


# ==========================================================================
# 4. THRESHOLD JOURNEY MATRIX & DECISION MATRIX
# ==========================================================================

def threshold_journey_matrix(
    person_ids,
    data,
    eer_threshold,
    higher_is_match=True,
    num_thresholds=1000,
):
    enrollment = {pid: data[pid]["enrollment"] for pid in person_ids}
    live_sets  = {pid: [emb for _, emb in data[pid]["live_sets"]] for pid in person_ids}

    # Calculate mean genuine score per person for the journey matrix
    person_gen_means = []
    for pid in person_ids:
        samples = live_sets[pid] if live_sets[pid] else [enrollment[pid]]
        e_norm = enrollment[pid] / (np.linalg.norm(enrollment[pid]) + 1e-12)
        norm_samples = [s / (np.linalg.norm(s) + 1e-12) for s in samples]
        mean_score = float(np.mean([np.dot(s_norm, e_norm) for s_norm in norm_samples]))
        person_gen_means.append(mean_score)

    all_scores = np.array(person_gen_means)
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)
    step_size = thresholds[1] - thresholds[0]

    rows = []
    for t in thresholds:
        decisions = ["ACCEPT" if s >= t else "REJECT" for s in person_gen_means]
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


def per_person_metrics(person_ids, cos_df, higher_is_match=True):
    rows   = []
    scores = cos_df.values

    for idx, pid in enumerate(person_ids):
        row             = scores[idx, :]
        genuine_score   = row[idx]
        impostor_scores = np.delete(row, idx)
        candidate_thresholds = np.linspace(row.min(), row.max(), 500)
        best_t, best_diff, best_far, best_frr = None, None, None, None

        for t in candidate_thresholds:
            far = np.mean(impostor_scores >= t) if higher_is_match else np.mean(impostor_scores <= t)
            frr = (1.0 if genuine_score < t else 0.0) if higher_is_match else (1.0 if genuine_score > t else 0.0)
            diff = abs(far - frr)
            if best_diff is None or diff < best_diff:
                best_diff, best_t, best_far, best_frr = diff, t, far, frr

        rows.append({
            "person_id":           pid,
            "genuine_score":       genuine_score,
            "mean_impostor_score": impostor_scores.mean(),
            "max_impostor_score":  impostor_scores.max() if higher_is_match else impostor_scores.min(),
            "local_threshold":     best_t,
            "local_FAR":           best_far,
            "local_FRR":           best_frr,
        })

    return pd.DataFrame(rows)


# ==========================================================================
# 5. MAIN ENTRY POINT
# ==========================================================================

def main(db_path, dataset, fusion_type, out_dir, template_type="raw"):
    os.makedirs(out_dir, exist_ok=True)

    data = load_fused_data(db_path, dataset, fusion_type, template_type=template_type)
    person_ids = sorted(data.keys())

    gen_cos, imp_cos, gen_euc, imp_euc, cos_df, euc_df = compute_fused_scores_and_matrices(person_ids, data)

    sep = "=" * 60
    print(f"\n{sep}\nEvaluating Fused Multimodal Embeddings ({fusion_type.upper()}, type={template_type.upper()})\n{sep}")
    print(f"  Persons         : {len(person_ids)}")
    print(f"  Genuine Pairs   : {len(gen_cos)}")
    print(f"  Impostor Pairs  : {len(imp_cos)}")
    print(f"  Embedding Vector: {data[person_ids[0]]['enrollment'].shape[0]}-D")

    # Matrices
    cos_df.to_csv(os.path.join(out_dir, "fused_cosine_similarity_matrix.csv"))
    euc_df.to_csv(os.path.join(out_dir, "fused_euclidean_distance_matrix.csv"))

    # Label matrix
    label_df = genuine_impostor_label_matrix(person_ids)
    label_df.to_csv(os.path.join(out_dir, "fused_genuine_impostor_label_matrix.csv"))

    # Cosine sweep + EER + decision matrix
    sweep_cos            = threshold_sweep(gen_cos, imp_cos, higher_is_match=True)
    eer_cos, eer_thr_cos = compute_eer(sweep_cos)

    sweep_cos.to_csv(os.path.join(out_dir, "fused_threshold_sweep_cosine.csv"), index=False)
    decision_matrix(cos_df, label_df, eer_thr_cos).to_csv(
        os.path.join(out_dir, "fused_decision_matrix_cosine.csv"))

    # Threshold Journey Matrix
    journey_df = threshold_journey_matrix(
        person_ids, data,
        eer_threshold=eer_thr_cos,
        higher_is_match=True,
        num_thresholds=1000,
    )
    journey_path = os.path.join(out_dir, "fused_threshold_journey_matrix.csv")
    journey_df.to_csv(journey_path, index=False)
    print(f"  Journey matrix saved: {journey_path} ({len(journey_df)} threshold steps x {len(person_ids)} persons)")

    # Euclidean sweep + EER + decision matrix
    sweep_euc            = threshold_sweep(gen_euc, imp_euc, higher_is_match=False)
    eer_euc, eer_thr_euc = compute_eer(sweep_euc)

    sweep_euc.to_csv(os.path.join(out_dir, "fused_threshold_sweep_euclidean.csv"), index=False)
    decision_matrix(euc_df, label_df, eer_thr_euc, higher_is_match=False).to_csv(
        os.path.join(out_dir, "fused_decision_matrix_euclidean.csv"))

    # Per-person metrics
    per_person_df = per_person_metrics(person_ids, cos_df)
    per_person_df.to_csv(os.path.join(out_dir, "fused_per_person_metrics.csv"), index=False)
    avg_person_threshold = per_person_df["local_threshold"].mean()

    # Console Summary
    print(f"\n{sep}\nFUSED EMBEDDING BENCHMARK RESULTS ({template_type.upper()})\n{sep}")
    print(f"  Genuine  scores : {len(gen_cos):>5}  mean={gen_cos.mean():.4f}")
    print(f"  Impostor scores : {len(imp_cos):>5}  mean={imp_cos.mean():.4f}")
    print(f"  Cosine    EER   = {eer_cos:.4f} ({eer_cos*100:.2f}%)  @ threshold = {eer_thr_cos:.4f}")
    print(f"  Euclidean EER   = {eer_euc:.4f} ({eer_euc*100:.2f}%)  @ threshold = {eer_thr_euc:.4f}")
    print(f"  Avg per-person threshold = {avg_person_threshold:.4f}")

    summary_df = pd.DataFrame([{
        "fusion_type":                     fusion_type,
        "template_type":                   template_type,
        "n_persons":                       len(person_ids),
        "genuine_pairs":                   len(gen_cos),
        "impostor_pairs":                  len(imp_cos),
        "genuine_mean":                    round(float(gen_cos.mean()), 6),
        "impostor_mean":                   round(float(imp_cos.mean()), 6),
        "cosine_EER":                      eer_cos,
        "cosine_EER_threshold":            eer_thr_cos,
        "euclidean_EER":                   eer_euc,
        "euclidean_EER_threshold":         eer_thr_euc,
        "avg_per_person_cosine_threshold": avg_person_threshold,
    }])
    summary_df.to_csv(os.path.join(out_dir, "fused_summary.csv"), index=False)

    print(f"\nAll fused evaluation matrices and tables saved under: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fused embeddings evaluation and threshold journey matrix generator."
    )
    parser.add_argument("--db",            type=str, default="database/biometric_final.db",
                        help="Path to SQLite database.")
    parser.add_argument("--dataset",       type=str, default="setA",
                        help="Dataset tag.")
    parser.add_argument("--fusion_type",   type=str, default="cbp",
                        help="Fusion type stored in database (e.g. cbp, concat_l2).")
    parser.add_argument("--template_type", type=str, default="biohash", choices=["raw", "biohash"],
                        help="Type of template to evaluate: 'raw' or 'biohash'.")
    parser.add_argument("--out_dir",       type=str, default="results/fused_eval",
                        help="Output directory for fused evaluation matrices.")
    args = parser.parse_args()

    main(args.db, args.dataset, args.fusion_type, args.out_dir, args.template_type)

