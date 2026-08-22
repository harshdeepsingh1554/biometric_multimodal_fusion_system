"""
Biometric Multimodal Evaluation - Matrix Generator
====================================================

Computes, per modality (face / iris / finger), independently:
    1. Person x Person cosine similarity matrix
    2. Person x Person euclidean distance matrix
    3. Genuine / Impostor score matrix (diagonal = genuine, off-diagonal = impostor)
    4. Decision matrix (TP / FN / TN / FP) at the EER threshold
    5. Threshold sweep table (threshold, FAR, FRR) for cosine AND euclidean
    6. Per-person local threshold, local FAR, local FRR, local EER estimate
    7. Global EER + EER threshold
    8. Average threshold recommendation per modality

NO FUSION IS DONE HERE. Each modality is evaluated completely independently.

--------------------------------------------------------------------------
SQLite DB FORMAT (source of embeddings)
--------------------------------------------------------------------------
Tables used:
  templates        -> averaged embedding per (person, dataset, trait)
                      used as the ENROLLMENT template
  image_embeddings -> all per-image embeddings per (person, dataset, trait)
                      the LAST image_index per person is used as the PROBE
                      (held-out sample, not averaged into the template)

Embeddings stored as float32 BLOB; recovered with:
    np.frombuffer(blob, dtype=np.float32)
--------------------------------------------------------------------------
"""

import os
import sqlite3
import argparse
import numpy as np
import pandas as pd

MODALITIES = ["face", "iris", "finger"]


# ==========================================================================
# 1. LOAD DATA FROM SQLITE
# ==========================================================================

def load_from_sqlite(db_path, dataset, template_type="raw"):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    if template_type == "biohash":
        tmpl_table = "biohash_embeddings" if "biohash_embeddings" in tables else "biohash_templates"
        live_table = "biohash_live_embeddings" if "biohash_live_embeddings" in tables else "biohash_image_embeddings"
    else:
        tmpl_table = "EMBEDDINGS" if "EMBEDDINGS" in tables else "templates"
        live_table = "live_embeddings" if "live_embeddings" in tables else "image_embeddings"

    if tmpl_table not in tables:
        raise ValueError(f"Table '{tmpl_table}' not found in {db_path}.")

    tmpl_rows = cur.execute(
        f"SELECT person_id, trait, embedding, dim FROM {tmpl_table} "
        "WHERE dataset=? ORDER BY person_id, trait",
        (dataset,)
    ).fetchall()

    live_rows = []
    if live_table in tables:
        cols = [c[1] for c in cur.execute(f"PRAGMA table_info({live_table})").fetchall()]
        idx_col = "live_index" if "live_index" in cols else "image_index"
        live_rows = cur.execute(
            f"SELECT person_id, trait, {idx_col}, embedding, dim "
            f"FROM {live_table} WHERE dataset=? ORDER BY person_id, trait, {idx_col}",
            (dataset,)
        ).fetchall()

    conn.close()

    if not tmpl_rows:
        raise ValueError(f"No templates found in '{tmpl_table}' for dataset={dataset!r}.")

    data = {}
    for pid, trait, blob, dim in tmpl_rows:
        data.setdefault(pid, {})
        emb = np.frombuffer(blob, dtype=np.float32).astype(np.float64).copy()
        data[pid].setdefault(trait, {})["enrollment"] = emb
        data[pid][trait]["live_sets"] = []

    for pid, trait, idx, blob, dim in live_rows:
        if pid in data and trait in data[pid]:
            emb = np.frombuffer(blob, dtype=np.float32).astype(np.float64).copy()
            data[pid][trait]["live_sets"].append((idx, emb))

    print(f"[DB] Loaded {len(data)} persons from '{tmpl_table}' & '{live_table}'  dataset={dataset!r}  type={template_type.upper()}")
    for trait in MODALITIES:
        n_with = sum(1 for p in data.values() if trait in p and p[trait].get("live_sets"))
        n_imgs = sum(len(p[trait].get("live_sets", [])) for p in data.values() if trait in p)
        print(f"  {trait}: {n_with} persons, {n_imgs} total live set embeddings")

    return data


def compute_modality_scores_and_matrices(person_ids, data, modality):
    enrollment = {pid: data[pid][modality]["enrollment"] for pid in person_ids}
    live_sets  = {pid: [emb for _, emb in data[pid][modality].get("live_sets", [])] for pid in person_ids}

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


def threshold_journey_matrix(
    person_ids,
    data,
    modality,
    eer_threshold,
    higher_is_match=True,
    num_thresholds=100,
):
    enrollment = {pid: data[pid][modality]["enrollment"] for pid in person_ids}
    live_sets  = {pid: [emb for _, emb in data[pid][modality].get("live_sets", [])] for pid in person_ids}

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
            if higher_is_match:
                far = np.mean(impostor_scores >= t)
                frr = 1.0 if genuine_score < t else 0.0
            else:
                far = np.mean(impostor_scores <= t)
                frr = 1.0 if genuine_score > t else 0.0
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
# 4. MAIN PIPELINE PER MODALITY
# ==========================================================================

def evaluate_modality(modality, data, out_dir):
    sep = "=" * 60
    print(f"\n{sep}\nEvaluating modality: {modality.upper()}\n{sep}")

    person_ids = sorted(pid for pid in data if modality in data[pid] and "enrollment" in data[pid][modality])
    mod_dir = os.path.join(out_dir, modality)
    os.makedirs(mod_dir, exist_ok=True)

    gen_cos, imp_cos, gen_euc, imp_euc, cos_df, euc_df = compute_modality_scores_and_matrices(person_ids, data, modality)

    print(f"  Persons         : {len(person_ids)}")
    print(f"  Genuine Pairs   : {len(gen_cos)}")
    print(f"  Impostor Pairs  : {len(imp_cos)}")

    # Similarity and distance matrices
    cos_df.to_csv(os.path.join(mod_dir, f"{modality}_cosine_similarity_matrix.csv"))
    euc_df.to_csv(os.path.join(mod_dir, f"{modality}_euclidean_distance_matrix.csv"))

    # Genuine/impostor label matrix
    label_df = genuine_impostor_label_matrix(person_ids)
    label_df.to_csv(os.path.join(mod_dir, f"{modality}_genuine_impostor_label_matrix.csv"))

    # Cosine: sweep + EER + decision matrix
    sweep_cos            = threshold_sweep(gen_cos, imp_cos, higher_is_match=True)
    eer_cos, eer_thr_cos = compute_eer(sweep_cos)
    sweep_cos.to_csv(os.path.join(mod_dir, f"{modality}_threshold_sweep_cosine.csv"), index=False)
    decision_matrix(cos_df, label_df, eer_thr_cos).to_csv(
        os.path.join(mod_dir, f"{modality}_decision_matrix_cosine.csv"))

    # Threshold journey matrix
    journey_df = threshold_journey_matrix(
        person_ids, data, modality,
        eer_threshold=eer_thr_cos,
        higher_is_match=True,
        num_thresholds=1000,
    )
    journey_path = os.path.join(mod_dir, f"{modality}_threshold_journey_matrix.csv")
    journey_df.to_csv(journey_path, index=False)
    print(f"  Journey matrix saved: {journey_path}  "
          f"({len(journey_df)} threshold steps x {len(person_ids)} persons)")

    # Euclidean: sweep + EER + decision matrix
    sweep_euc            = threshold_sweep(gen_euc, imp_euc, higher_is_match=False)
    eer_euc, eer_thr_euc = compute_eer(sweep_euc)
    sweep_euc.to_csv(os.path.join(mod_dir, f"{modality}_threshold_sweep_euclidean.csv"), index=False)
    decision_matrix(euc_df, label_df, eer_thr_euc, higher_is_match=False).to_csv(
        os.path.join(mod_dir, f"{modality}_decision_matrix_euclidean.csv"))

    # Per-person metrics
    per_person_df = per_person_metrics(person_ids, cos_df)
    per_person_df.to_csv(os.path.join(mod_dir, f"{modality}_per_person_metrics.csv"), index=False)
    avg_person_threshold = per_person_df["local_threshold"].mean()

    print(f"  Genuine  scores : {len(gen_cos):>5}  mean={gen_cos.mean():.4f}")
    print(f"  Impostor scores : {len(imp_cos):>5}  mean={imp_cos.mean():.4f}")
    print(f"  Cosine    EER = {eer_cos:.4f}  @ threshold = {eer_thr_cos:.4f}")
    print(f"  Euclidean EER = {eer_euc:.4f}  @ threshold = {eer_thr_euc:.4f}")
    print(f"  Avg per-person cosine threshold = {avg_person_threshold:.4f}")

    return {
        "modality":                        modality,
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
    }


# ==========================================================================
# 5. ENTRY POINT
# ==========================================================================

def main(db_path, dataset, out_dir, template_type="biohash"):
    os.makedirs(out_dir, exist_ok=True)
    data = load_from_sqlite(db_path, dataset, template_type=template_type)

    summary_rows = []
    for modality in MODALITIES:
        summary_rows.append(evaluate_modality(modality, data, out_dir))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary_all_modalities.csv"), index=False)

    overall_avg_thr = summary_df["cosine_EER_threshold"].mean()
    overall_avg_eer = summary_df["cosine_EER"].mean()

    sep = "=" * 60
    print(f"\n{sep}\nOVERALL SUMMARY (template_type={template_type.upper()})\n{sep}")
    print(summary_df.to_string(index=False))
    print(f"\nAverage cosine EER threshold: {overall_avg_thr:.4f}")
    print(f"Average cosine EER:           {overall_avg_eer:.4f}")

    with open(os.path.join(out_dir, "overall_summary.txt"), "w") as f:
        f.write(summary_df.to_string(index=False))
        f.write(f"\n\nAverage cosine EER threshold: {overall_avg_thr:.4f}")
        f.write(f"\nAverage cosine EER:           {overall_avg_eer:.4f}\n")

    print(f"\nAll results saved under: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Biometric evaluation matrix generator - reads from SQLite DB."
    )
    parser.add_argument("--db",            type=str, default="database/biometric_final.db",
                        help="Path to SQLite biometric database.")
    parser.add_argument("--dataset",       type=str, default="setA",
                        help="Dataset tag (matches --dataset used during enrollment).")
    parser.add_argument("--template_type", type=str, default="biohash", choices=["raw", "biohash"],
                        help="Type of template to evaluate: 'raw' or 'biohash'.")
    parser.add_argument("--out_dir",       type=str, default="results/unfused_eval",
                        help="Folder to save all result matrices/tables.")
    args = parser.parse_args()

    main(args.db, args.dataset, args.out_dir, args.template_type)