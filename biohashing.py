"""
biohashing.py — Traditional BioHashing for Biometric Template Protection
========================================================================

Implements Traditional BioHashing (Jin et al., 2004) for cancelable biometrics:
    1. Loads biometric embeddings from SQLite (templates and image_embeddings).
    2. Generates an orthonormal random matrix R (size D x M) using a secret key seed.
    3. Projects embedding vector x (dim D): y = x * R  (dim M).
    4. Binarizes via threshold tau (0): b = sign(y) -> {-1, +1}^M.
    5. Normalizes: b_norm = b / sqrt(M)  (Hamming distance maps 1:1 to Cosine similarity).
    6. Saves secret keys into a JSON key file (e.g. database/biohash_keys.json).
    7. Stores BioHashed embeddings back into SQLite tables:
         - biohash_templates         (BioHashed enrolled templates)
         - biohash_image_embeddings   (BioHashed probe embeddings)

Usage:
    python biohashing.py --db database/biometric.db --dataset setA --hash_dim 512 --key_seed 2026
"""

import argparse
import hashlib
import json
import os
import sqlite3
import numpy as np


# ==========================================================================
# 1. TRADITIONAL BIOHASHING ENGINE
# ==========================================================================

class BioHasher:
    def __init__(self, input_dim: int, hash_dim: int, seed: int = 2026):
        """
        Traditional BioHasher.

        Parameters
        ----------
        input_dim : Dimension of input feature vector D (e.g. 512, 1536, 2048)
        hash_dim  : Code length of output BioHash M (e.g. 512, 1024)
        seed      : Secret key seed used to generate orthonormal random matrix R
        """
        if hash_dim > input_dim:
            raise ValueError(
                f"hash_dim ({hash_dim}) cannot exceed input_dim ({input_dim}): "
                f"QR can only produce min(input_dim, hash_dim) orthonormal columns."
            )

        self.input_dim = input_dim
        self.hash_dim  = hash_dim
        self.seed      = seed

        # Generate orthonormal random projection matrix R (D x M) via QR decomposition
        rng = np.random.RandomState(seed)
        gaussian_mat = rng.randn(input_dim, hash_dim)
        # QR decomposition guarantees orthonormal columns R^T R = I_M
        q, _ = np.linalg.qr(gaussian_mat)
        self.R = q[:, :hash_dim].astype(np.float64)  # (D, M)

        assert self.R.shape == (input_dim, hash_dim), (
            f"Projection matrix shape mismatch: expected {(input_dim, hash_dim)}, "
            f"got {self.R.shape}."
        )

    def generate_biohash(self, x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Apply BioHashing to a single vector or batch of vectors x.

        Steps:
            1. Projection: y = x * R                    (dim M)
            2. Thresholding: b = +1 if y >= 0 else -1    (bipolar code)
            3. Normalization: b / sqrt(M)               (unit L2 norm)

        Returns
        -------
        b_norm : np.ndarray (float32, dim M)
        """
        x_vec = np.array(x, dtype=np.float64)
        if x_vec.ndim == 1:
            x_vec = x_vec.reshape(1, -1)

        # Step 1: Random projection
        y = np.dot(x_vec, self.R)  # (1, M)

        # Step 2: Thresholding / Binarization (Bipolar {-1, +1})
        b = np.where(y >= threshold, 1.0, -1.0)

        # Step 3: L2 normalize so Cosine similarity = 1 - 2*Hamming/M
        b_norm = b / np.sqrt(self.hash_dim)
        return b_norm.squeeze(0).astype(np.float32)


# ==========================================================================
# 2. DATABASE HELPER
# ==========================================================================

def create_biohash_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS biohash_embeddings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   TEXT    NOT NULL,
            dataset     TEXT    NOT NULL,
            trait       TEXT    NOT NULL,
            hash_dim    INTEGER NOT NULL,
            embedding   BLOB    NOT NULL,
            dim         INTEGER NOT NULL,
            UNIQUE(person_id, dataset, trait, hash_dim)
        );

        CREATE TABLE IF NOT EXISTS biohash_image_embeddings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id    TEXT    NOT NULL,
            dataset      TEXT    NOT NULL,
            trait        TEXT    NOT NULL,
            hash_dim     INTEGER NOT NULL,
            image_index  INTEGER NOT NULL,
            embedding    BLOB    NOT NULL,
            dim          INTEGER NOT NULL,
            UNIQUE(person_id, dataset, trait, hash_dim, image_index)
        );

        CREATE TABLE IF NOT EXISTS biohash_live_embeddings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id    TEXT    NOT NULL,
            dataset      TEXT    NOT NULL,
            live_index   INTEGER NOT NULL,
            trait        TEXT    NOT NULL,
            hash_dim     INTEGER NOT NULL,
            embedding    BLOB    NOT NULL,
            dim          INTEGER NOT NULL,
            UNIQUE(person_id, dataset, live_index, trait, hash_dim)
        );

        CREATE INDEX IF NOT EXISTS idx_bio_tmpl ON biohash_embeddings(person_id, dataset, trait, hash_dim);
        CREATE INDEX IF NOT EXISTS idx_bio_img  ON biohash_image_embeddings(person_id, dataset, trait, hash_dim, image_index);
        CREATE INDEX IF NOT EXISTS idx_bio_live ON biohash_live_embeddings(person_id, dataset, live_index, trait, hash_dim);
    """)
    conn.commit()


# ==========================================================================
# 3. MAIN PIPELINE
# ==========================================================================

def run_biohashing(
    db_path: str,
    dataset: str,
    hash_dim: int = 512,
    key_seed: int = 2026,
    keys_out_file: str = "database/biohash_keys.json",
):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    create_biohash_tables(conn)
    cur = conn.cursor()

    keys_data = {
        "dataset": dataset,
        "hash_dim": hash_dim,
        "global_seed": key_seed,
        "modalities": {},
    }

    print(f"\n{'='*60}\nRunning Traditional BioHashing\n{'='*60}")
    print(f"  Database : {db_path}")
    print(f"  Dataset  : {dataset}")
    print(f"  Hash Dim : {hash_dim}")
    print(f"  Seed     : {key_seed}")

    # =========================================================================
    # A. BIOHASH AVERAGE ENROLLED TEMPLATES (`EMBEDDINGS`)
    # =========================================================================

    tmpl_rows = cur.execute(
        "SELECT person_id, trait, embedding, dim FROM EMBEDDINGS WHERE dataset=? ORDER BY person_id, trait",
        (dataset,)
    ).fetchall()

    hasher_cache = {}

    def get_hasher(input_dim: int, trait_name: str) -> BioHasher:
        if (input_dim, trait_name) not in hasher_cache:
            # Deterministic per-trait seed derived from master key_seed via SHA-256
            # (Python's built-in hash() is randomized per-process for strings and
            # must never be used for reproducible seeding across runs.)
            seed_string = f"{key_seed}_{trait_name}"
            hash_bytes = hashlib.sha256(seed_string.encode()).digest()
            trait_seed = int.from_bytes(hash_bytes[:4], 'big') & 0x7FFFFFFF

            hasher = BioHasher(input_dim=input_dim, hash_dim=hash_dim, seed=trait_seed)
            hasher_cache[(input_dim, trait_name)] = hasher
            keys_data["modalities"][trait_name] = {
                "input_dim": input_dim,
                "hash_dim": hash_dim,
                "seed": trait_seed,
            }
        return hasher_cache[(input_dim, trait_name)]

    # BioHash single-modality average templates
    bio_tmpl_cnt = 0
    for pid, trait, blob, dim in tmpl_rows:
        emb = np.frombuffer(blob, dtype=np.float32)
        hasher = get_hasher(dim, trait)
        bio_emb = hasher.generate_biohash(emb)

        cur.execute(
            "INSERT OR REPLACE INTO biohash_embeddings "
            "(person_id, dataset, trait, hash_dim, embedding, dim) VALUES (?,?,?,?,?,?)",
            (pid, dataset, trait, hash_dim, bio_emb.tobytes(), hash_dim)
        )
        bio_tmpl_cnt += 1

    print(f"[OK] BioHashed Enrolled Templates: {bio_tmpl_cnt} total stored (dim={hash_dim}).")

    # =========================================================================
    # B. BIOHASH LIVE SET EMBEDDINGS (`live_embeddings`)
    # =========================================================================

    # Check if live_embeddings table exists and fetch rows
    tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    bio_live_cnt = 0
    if "live_embeddings" in tables:
        live_rows = cur.execute(
            "SELECT person_id, live_index, trait, embedding, dim "
            "FROM live_embeddings WHERE dataset=? ORDER BY person_id, live_index, trait",
            (dataset,)
        ).fetchall()

        for pid, l_idx, trait, blob, dim in live_rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            hasher = get_hasher(dim, trait)
            bio_emb = hasher.generate_biohash(emb)

            cur.execute(
                "INSERT OR REPLACE INTO biohash_live_embeddings "
                "(person_id, dataset, live_index, trait, hash_dim, embedding, dim) VALUES (?,?,?,?,?,?,?)",
                (pid, dataset, l_idx, trait, hash_dim, bio_emb.tobytes(), hash_dim)
            )
            bio_live_cnt += 1

        print(f"[OK] BioHashed Live Set Embeddings: {bio_live_cnt} total stored (dim={hash_dim}).")
    else:
        print("[WARN] Table 'live_embeddings' not found in database. Skipped BioHashing live sets.")

    conn.commit()
    conn.close()

    # Save secret projection keys metadata to JSON file
    os.makedirs(os.path.dirname(os.path.abspath(keys_out_file)), exist_ok=True)
    with open(keys_out_file, "w") as f:
        json.dump(keys_data, f, indent=2)

    print(f"[OK] BioHash Keys saved to: {keys_out_file}")
    print(f"[DONE] Traditional BioHashing process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traditional BioHashing template protection for multimodal biometrics."
    )
    parser.add_argument("--db",       default="database/biometric.db", help="Path to SQLite database.")
    parser.add_argument("--dataset",  default="setA",                help="Dataset tag.")
    parser.add_argument("--hash_dim", type=int, default=512,         help="BioHash code length M (e.g. 512).")
    parser.add_argument("--key_seed", type=int, default=2026,        help="Secret key seed for projection matrix.")
    parser.add_argument("--keys_out", default="database/biohash_keys.json", help="Path to output JSON keys file.")
    args = parser.parse_args()

    run_biohashing(args.db, args.dataset, args.hash_dim, args.key_seed, args.keys_out)