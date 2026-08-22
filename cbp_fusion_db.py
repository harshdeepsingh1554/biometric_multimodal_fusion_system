"""
cbp_fusion_db.py — Generalized Compact Bilinear Fusion (CBP)
============================================================

Implements Generalized Compact Bilinear Fusion (GCBF) via Count Sketch + FFT:
    Stage 1: Fuse Face (512-D) and Iris (512-D) -> Intermediate Fused (output_dim)
    Stage 2: Fuse Intermediate (output_dim) and Fingerprint (512-D) -> Final Fused Feature (output_dim)

Post-Processing:
    - Signed Square Root: z' = sign(z) * sqrt(|z|)
    - L2 Normalization:   z_final = z' / ||z'||_2

Stores fused embeddings into SQLite tables:
    - fused_templates         (fusion_type='cbp')
    - fused_image_embeddings   (fusion_type='cbp')

Usage:
    python cbp_fusion_db.py --db database/biometric.db --dataset setA --output_dim 2048
"""

import argparse
import os
import sqlite3
import torch
import torch.nn as nn
from torch.fft import fft, ifft
import numpy as np


# ==========================================================================
# 1. GENERALIZED COMPACT BILINEAR FUSION CLASS
# ==========================================================================

class GeneralizedCompactBilinearFusion(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int, output_dim: int):
        """
        Generalized Compact Bilinear Fusion (GCBF) module.

        Parameters
        ----------
        input_dim1 : Dimension of first input modality (e.g. 512)
        input_dim2 : Dimension of second input modality (e.g. 512 or intermediate dim)
        output_dim : Dimension of fused output sketch (e.g. 2048 or 4096)
        """
        super(GeneralizedCompactBilinearFusion, self).__init__()
        self.output_dim = output_dim

        # Fixed random projections for Count Sketch (hashing and signs)
        self.register_buffer('sketch1', torch.randint(0, output_dim, (input_dim1,)))
        self.register_buffer('sketch2', torch.randint(0, output_dim, (input_dim2,)))
        self.register_buffer('sign1', 2 * torch.randint(0, 2, (input_dim1,)) - 1)
        self.register_buffer('sign2', 2 * torch.randint(0, 2, (input_dim2,)) - 1)

        # Learnable weights for each modality
        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))

    def count_sketch(self, x: torch.Tensor, sketch: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """Apply Count Sketch: compress high-dim vector into low-dim sketch."""
        # x: (batch_size, input_dim)
        batch_size = x.size(0)
        s = torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)
        # index_add_ efficiently accumulates values into their hashed buckets
        s.index_add_(1, sketch, x * sign)
        return s

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 1. Apply weights
        x1_weighted = x1 * self.weight1
        x2_weighted = x2 * self.weight2

        # 2. Count Sketch (dimensionality projection)
        s1 = self.count_sketch(x1_weighted, self.sketch1, self.sign1)
        s2 = self.count_sketch(x2_weighted, self.sketch2, self.sign2)

        # 3. Fast Fourier Transform (move to frequency domain)
        f1 = fft(s1)
        f2 = fft(s2)

        # 4. Element-wise multiplication in frequency domain (approximates convolution / outer product)
        f_product = f1 * f2

        # 5. Inverse FFT and take real part
        fused = ifft(f_product).real
        return fused


# ==========================================================================
# 2. POST-PROCESSING: SIGNED SQUARE ROOT + L2 NORMALIZATION
# ==========================================================================

def post_process_cbp(fused_tensor: torch.Tensor) -> np.ndarray:
    """
    Standard CBP post-processing (Gao et al.):
        1. Signed Square Root: z = sign(fused) * sqrt(|fused|)
        2. L2 Normalization:   z / ||z||_2
    """
    z = torch.sign(fused_tensor) * torch.sqrt(torch.abs(fused_tensor) + 1e-12)
    norm = torch.norm(z, dim=-1, keepdim=True)
    z_norm = z / torch.clamp(norm, min=1e-12)
    return z_norm.detach().cpu().numpy().squeeze(0).astype(np.float32)


# ==========================================================================
# 3. DATABASE HELPER
# ==========================================================================

def create_fused_tables(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS biohash_embedding_fused (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   TEXT    NOT NULL,
            dataset     TEXT    NOT NULL,
            fusion_type TEXT    DEFAULT 'cbp',
            embedding   BLOB    NOT NULL,
            dim         INTEGER NOT NULL,
            UNIQUE(person_id, dataset, fusion_type)
        );

        CREATE TABLE IF NOT EXISTS biohashed_livefused (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id    TEXT    NOT NULL,
            dataset      TEXT    NOT NULL,
            fusion_type  TEXT    DEFAULT 'cbp',
            live_index   INTEGER NOT NULL,
            embedding    BLOB    NOT NULL,
            dim          INTEGER NOT NULL,
            UNIQUE(person_id, dataset, fusion_type, live_index)
        );

        CREATE INDEX IF NOT EXISTS idx_bio_fused_tmpl ON biohash_embedding_fused(person_id, dataset);
        CREATE INDEX IF NOT EXISTS idx_bio_fused_live ON biohashed_livefused(person_id, dataset, live_index);
    """)
    conn.commit()


# ==========================================================================
# 4. MAIN FUSION PROCESS
# ==========================================================================

def process_cbp_fusion(db_path: str, dataset: str, output_dim: int = 2048, seed: int = 2026):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    torch.manual_seed(seed)

    EMBED_DIM = 512

    # Initialize two GCBF modules as specified by protocol
    # Stage 1: Fuse Face (512) and Iris (512) -> Intermediate (output_dim)
    fusion_face_iris = GeneralizedCompactBilinearFusion(EMBED_DIM, EMBED_DIM, output_dim)

    # Stage 2: Fuse Intermediate (output_dim) and Fingerprint (512) -> Final Fused Feature (output_dim)
    fusion_fused_finger = GeneralizedCompactBilinearFusion(output_dim, EMBED_DIM, output_dim)

    fusion_face_iris.eval()
    fusion_fused_finger.eval()

    conn = sqlite3.connect(db_path)
    create_fused_tables(conn)
    cur = conn.cursor()

    # Check available table names
    db_tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    # =========================================================================
    # A. FUSE AVERAGE BIOHASH EMBEDDINGS -> biohash_embedding_fused
    # =========================================================================
    tmpl_table = "biohash_embeddings" if "biohash_embeddings" in db_tables else "biohash_templates"
    tmpl_rows = cur.execute(
        f"SELECT person_id, trait, embedding, dim FROM {tmpl_table} "
        "WHERE dataset=? ORDER BY person_id, trait",
        (dataset,)
    ).fetchall()

    if not tmpl_rows:
        print(f"[WARN] No templates found in '{tmpl_table}' for dataset='{dataset}' in {db_path}")
    else:
        person_templates = {}
        for pid, trait, blob, dim in tmpl_rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            person_templates.setdefault(pid, {})[trait] = emb

        fused_tmpl_cnt = 0
        with torch.no_grad():
            for pid, traits in person_templates.items():
                if not ("face" in traits and "finger" in traits and "iris" in traits):
                    print(f"[WARN] Skipping {pid}: missing modality templates.")
                    continue

                t_face   = torch.from_numpy(traits["face"]).unsqueeze(0)   # (1, 512)
                t_finger = torch.from_numpy(traits["finger"]).unsqueeze(0) # (1, 512)
                t_iris   = torch.from_numpy(traits["iris"]).unsqueeze(0)   # (1, 512)

                # Stage 1: Fuse Face and Iris
                fused_fi = fusion_face_iris(t_face, t_iris)                # (1, output_dim)

                # Stage 2: Fuse intermediate result with Fingerprint
                fused_final_raw = fusion_fused_finger(fused_fi, t_finger) # (1, output_dim)

                # Post-process: Signed Square Root + L2 Normalization
                fused_vec = post_process_cbp(fused_final_raw)             # (output_dim,)

                cur.execute(
                    "INSERT OR REPLACE INTO biohash_embedding_fused "
                    "(person_id, dataset, fusion_type, embedding, dim) VALUES (?,?,'cbp',?,?)",
                    (pid, dataset, fused_vec.tobytes(), fused_vec.shape[0])
                )
                fused_tmpl_cnt += 1

        print(f"[OK] Fused Average BioHash Templates: {fused_tmpl_cnt} stored in biohash_embedding_fused (dim={output_dim}).")

    # =========================================================================
    # B. FUSE LIVE BIOHASH EMBEDDINGS -> biohashed_livefused
    # =========================================================================
    if "biohash_live_embeddings" in db_tables:
        live_rows = cur.execute(
            "SELECT person_id, live_index, trait, embedding, dim "
            "FROM biohash_live_embeddings WHERE dataset=? ORDER BY person_id, live_index, trait",
            (dataset,)
        ).fetchall()

        person_live = {}
        for pid, l_idx, trait, blob, dim in live_rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            person_live.setdefault(pid, {}).setdefault(l_idx, {})[trait] = emb

        fused_live_cnt = 0
        with torch.no_grad():
            for pid, live_dict in person_live.items():
                for l_idx, traits in live_dict.items():
                    if not ("face" in traits and "finger" in traits and "iris" in traits):
                        continue

                    l_face   = torch.from_numpy(traits["face"]).unsqueeze(0)
                    l_finger = torch.from_numpy(traits["finger"]).unsqueeze(0)
                    l_iris   = torch.from_numpy(traits["iris"]).unsqueeze(0)

                    # Stage 1: Face + Iris
                    l_fused_fi = fusion_face_iris(l_face, l_iris)

                    # Stage 2: Intermediate + Fingerprint
                    l_fused_raw = fusion_fused_finger(l_fused_fi, l_finger)

                    # Post-process
                    l_fused_vec = post_process_cbp(l_fused_raw)

                    cur.execute(
                        "INSERT OR REPLACE INTO biohashed_livefused "
                        "(person_id, dataset, fusion_type, live_index, embedding, dim) VALUES (?,?,'cbp',?,?,?)",
                        (pid, dataset, l_idx, l_fused_vec.tobytes(), l_fused_vec.shape[0])
                    )
                    fused_live_cnt += 1

        print(f"[OK] Fused Live BioHash Embeddings: {fused_live_cnt} stored in biohashed_livefused (dim={output_dim}).")
    else:
        print("[WARN] Table 'biohash_live_embeddings' not found in DB. Skipped live biohash fusion.")

    conn.commit()
    conn.close()
    print(f"[DONE] CBP Fusion process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generalized Compact Bilinear Fusion (CBP) for Multimodal Biometrics."
    )
    parser.add_argument("--db",         default="database/biometric_final.db", help="Path to SQLite database.")
    parser.add_argument("--dataset",    default="setA",                     help="Dataset tag.")
    parser.add_argument("--output_dim", type=int, default=2048,             help="Fused sketch output dimension (e.g. 2048).")
    parser.add_argument("--seed",       type=int, default=2026,             help="Random seed for Count Sketch projections.")
    args = parser.parse_args()

    process_cbp_fusion(args.db, args.dataset, args.output_dim, args.seed)
