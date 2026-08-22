"""
concat_fusion_db.py — Feature Concatenation & Database Persistence
===================================================================

Reads single-modality embeddings (face, finger, iris) from SQLite,
concatenates them into a unified 1536-dimensional L2-normalized vector,
and saves the fused embeddings back into SQLite tables:
    - fused_templates         (1536-dim enrolled template per person)
    - fused_image_embeddings   (1536-dim probe embedding per person)

Storage format: float32 BLOB (zero precision loss).

Usage:
    python concat_fusion_db.py --db database/biometric.db --dataset setA
"""

import argparse
import os
import sqlite3
import numpy as np


def create_fused_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS fused_templates (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id  TEXT    NOT NULL,
            dataset    TEXT    NOT NULL,
            fusion_type TEXT   DEFAULT 'concat_l2',
            embedding  BLOB    NOT NULL,
            dim        INTEGER NOT NULL,
            UNIQUE(person_id, dataset, fusion_type)
        );

        CREATE TABLE IF NOT EXISTS fused_image_embeddings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id    TEXT    NOT NULL,
            dataset      TEXT    NOT NULL,
            fusion_type  TEXT    DEFAULT 'concat_l2',
            probe_index  INTEGER NOT NULL,
            embedding    BLOB    NOT NULL,
            dim          INTEGER NOT NULL,
            UNIQUE(person_id, dataset, fusion_type, probe_index)
        );

        CREATE INDEX IF NOT EXISTS idx_fused_tmpl ON fused_templates(person_id, dataset);
        CREATE INDEX IF NOT EXISTS idx_fused_img  ON fused_image_embeddings(person_id, dataset, probe_index);
    """)
    conn.commit()


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2 normalize a 1D vector."""
    norm = np.linalg.norm(v)
    return (v / max(norm, 1e-12)).astype(np.float32)


def process_concatenation(db_path: str, dataset: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    create_fused_tables(conn)
    cur = conn.cursor()

    # =========================================================================
    # 1. CONCATENATE ENROLLED TEMPLATES
    # =========================================================================
    tmpl_rows = cur.execute(
        "SELECT person_id, trait, embedding, dim FROM templates "
        "WHERE dataset=? ORDER BY person_id, trait",
        (dataset,)
    ).fetchall()

    if not tmpl_rows:
        raise ValueError(f"No templates found for dataset='{dataset}' in {db_path}")

    person_templates = {}
    for pid, trait, blob, dim in tmpl_rows:
        emb = np.frombuffer(blob, dtype=np.float32).copy()
        person_templates.setdefault(pid, {})[trait] = emb

    fused_tmpl_cnt = 0
    for pid, traits in person_templates.items():
        if not ("face" in traits and "finger" in traits and "iris" in traits):
            print(f"[WARN] Skipping {pid}: missing one or more modality templates.")
            continue

        # Concatenate face + finger + iris (512 + 512 + 512 = 1536-D)
        v_face   = traits["face"]
        v_finger = traits["finger"]
        v_iris   = traits["iris"]

        fused_raw = np.concatenate([v_face, v_finger, v_iris], axis=0)
        fused_vec = normalize_vector(fused_raw)

        cur.execute(
            "INSERT OR REPLACE INTO fused_templates "
            "(person_id, dataset, fusion_type, embedding, dim) VALUES (?,?,'concat_l2',?,?)",
            (pid, dataset, fused_vec.tobytes(), fused_vec.shape[0])
        )
        fused_tmpl_cnt += 1

    print(f"[OK] Fused Templates: {fused_tmpl_cnt} persons enrolled (dim={1536}).")

    # =========================================================================
    # 2. CONCATENATE PROBE IMAGE EMBEDDINGS
    # =========================================================================
    img_rows = cur.execute(
        "SELECT person_id, trait, image_index, embedding, dim "
        "FROM image_embeddings WHERE dataset=? ORDER BY person_id, trait, image_index",
        (dataset,)
    ).fetchall()

    person_images = {}
    for pid, trait, img_idx, blob, dim in img_rows:
        emb = np.frombuffer(blob, dtype=np.float32).copy()
        person_images.setdefault(pid, {}).setdefault(trait, {})[img_idx] = emb

    fused_img_cnt = 0
    for pid, traits in person_images.items():
        if not ("face" in traits and "finger" in traits and "iris" in traits):
            continue

        # Align probe indices across modalities
        face_indices   = sorted(traits["face"].keys())
        finger_indices = sorted(traits["finger"].keys())
        iris_indices   = sorted(traits["iris"].keys())

        # Determine number of aligned probes (e.g. minimum available per trait)
        num_probes = min(len(face_indices), len(finger_indices), len(iris_indices))

        for p_idx in range(num_probes):
            f_emb  = traits["face"][face_indices[p_idx]]
            fg_emb = traits["finger"][finger_indices[p_idx]]
            ir_emb = traits["iris"][iris_indices[p_idx]]

            fused_probe_raw = np.concatenate([f_emb, fg_emb, ir_emb], axis=0)
            fused_probe_vec = normalize_vector(fused_probe_raw)

            cur.execute(
                "INSERT OR REPLACE INTO fused_image_embeddings "
                "(person_id, dataset, fusion_type, probe_index, embedding, dim) VALUES (?,?,'concat_l2',?,?,?)",
                (pid, dataset, p_idx, fused_probe_vec.tobytes(), fused_probe_vec.shape[0])
            )
            fused_img_cnt += 1

    conn.commit()
    conn.close()

    print(f"[OK] Fused Image Embeddings: {fused_img_cnt} total probe vectors stored.")
    print(f"[DONE] Successfully updated SQLite DB with concatenated embeddings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate face, finger, and iris embeddings and store in SQLite DB."
    )
    parser.add_argument("--db",      default="database/biometric.db", help="Path to SQLite database.")
    parser.add_argument("--dataset", default="setA",                help="Dataset tag.")
    args = parser.parse_args()

    process_concatenation(args.db, args.dataset)
