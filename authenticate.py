"""
authenticate.py -- Multimodal Biometric Authentication System
=============================================================

Two-stage verification pipeline:
  Stage 1 -- Trait-Level BioHash Matching
    User provides 1 or 2 biometric traits (face / finger / iris image paths).
    Each probe trait is extracted -> biohashed -> compared against all enrolled
    biohash_embeddings in the database (1:N identification).
    The best-matching person_id is found (or the user-supplied enroll_id is used
    for 1:1 verification shortcut).

  Stage 2 -- Fused Embedding Verification
    Any missing third trait is fetched from the database for the candidate identity.
    All three biohashed traits are CBP-fused -> compared against the enrolled
    biohash_embedding_fused vector (and also 1:N across all enrolled fused vectors).
    ACCESS GRANTED if the fused similarity exceeds the configured threshold.

Usage examples:
    # Provide face + finger, let system recover iris from DB
    python authenticate.py --face probe_face.jpg --finger probe_finger.jpg --enroll-id Person_001

    # Provide all three traits, do 1:N search
    python authenticate.py --face p.jpg --finger f.jpg --iris i.jpg

    # Single trait probe
    python authenticate.py --face probe_face.jpg --enroll-id Person_045
"""

import os
import sys
import json
import argparse
import logging
import sqlite3
import textwrap
import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft, ifft
from datetime import datetime

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("authenticate")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extractors.face_extractor import FaceExtractor
from extractors.finger_extractor import FingerprintExtractor
from extractors.iris_extractor import IrisExtractor
from biohashing import BioHasher
from cbp_fusion_db import GeneralizedCompactBilinearFusion, post_process_cbp

DEFAULT_DB       = os.path.join(PROJECT_ROOT, "database", "biometric_final.db")
DEFAULT_KEY_FILE = os.path.join(PROJECT_ROOT, "database", "biohash_keys.json")
DATASET          = "setA"
HASH_DIM         = 512
OUTPUT_DIM       = 512
FUSION_SEED      = 2026
# Default per-trait thresholds (overridable via CLI args)
DEFAULT_THR = {
    "face"  : 0.30,
    "finger": 0.30,
    "iris"  : 0.20,
    "fused" : 0.0337,
}
TRAITS = ["face", "finger", "iris"]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ===========================================================================
# 1. HELPERS
# ===========================================================================

def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_biohash_keys(key_file):
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"BioHash key file not found: {key_file}")
    with open(key_file) as f:
        return json.load(f)


def build_biohashers(key_data):
    hashers = {}
    for trait, cfg in key_data["modalities"].items():
        hashers[trait] = BioHasher(
            input_dim=cfg["input_dim"],
            hash_dim=cfg["hash_dim"],
            seed=cfg["seed"],
        )
    return hashers


def build_cbp_modules(output_dim=OUTPUT_DIM, seed=FUSION_SEED):
    torch.manual_seed(seed)
    fi_fuser  = GeneralizedCompactBilinearFusion(HASH_DIM, HASH_DIM, output_dim)
    ff_fuser  = GeneralizedCompactBilinearFusion(output_dim, HASH_DIM, output_dim)
    fi_fuser.eval()
    ff_fuser.eval()
    return fi_fuser, ff_fuser


def cbp_fuse_three(face_bh, iris_bh, finger_bh, fi_fuser, ff_fuser):
    with torch.no_grad():
        t_face   = torch.from_numpy(face_bh).unsqueeze(0)
        t_iris   = torch.from_numpy(iris_bh).unsqueeze(0)
        t_finger = torch.from_numpy(finger_bh).unsqueeze(0)
        fused_fi  = fi_fuser(t_face, t_iris)
        fused_all = ff_fuser(fused_fi, t_finger)
    return post_process_cbp(fused_all)


# ===========================================================================
# 2. DATABASE ACCESS LAYER
# ===========================================================================

class BiometricDB:
    def __init__(self, db_path, dataset=DATASET):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        self.conn    = sqlite3.connect(db_path, check_same_thread=False)
        self.dataset = dataset

    def load_all_biohash_embeddings(self):
        rows = self.conn.execute(
            "SELECT person_id, trait, embedding FROM biohash_embeddings WHERE dataset=?",
            (self.dataset,)
        ).fetchall()
        db = {}
        for pid, trait, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            db.setdefault(pid, {})[trait] = emb
        return db

    def load_biohash_embedding(self, person_id, trait):
        row = self.conn.execute(
            "SELECT embedding FROM biohash_embeddings WHERE person_id=? AND dataset=? AND trait=?",
            (person_id, self.dataset, trait)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def load_all_fused_embeddings(self):
        rows = self.conn.execute(
            "SELECT person_id, embedding FROM biohash_embedding_fused WHERE dataset=?",
            (self.dataset,)
        ).fetchall()
        return {pid: np.frombuffer(blob, dtype=np.float32).copy() for pid, blob in rows}

    def load_fused_embedding(self, person_id):
        row = self.conn.execute(
            "SELECT embedding FROM biohash_embedding_fused WHERE person_id=? AND dataset=?",
            (person_id, self.dataset)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def close(self):
        self.conn.close()


# ===========================================================================
# 3. EXTRACTOR LOADER
# ===========================================================================

_extractors_cache = {}

def get_extractors(use_gpu=False):
    if _extractors_cache:
        return _extractors_cache
    print(f"{CYAN}[INFO]{RESET} Loading biometric extractors (first run may take a moment)...")
    _extractors_cache["face"] = FaceExtractor(
        model_path=os.path.join(PROJECT_ROOT, "weights/face/w600k_r50.onnx"),
        detector_path=os.path.join(PROJECT_ROOT, "weights/face/det_10g.onnx"),
        use_gpu_detector=use_gpu,   # FaceExtractor uses use_gpu_detector, not use_gpu
    )
    _extractors_cache["finger"] = FingerprintExtractor(
        model_path=os.path.join(PROJECT_ROOT, "weights/finger/finger_extractor_best.pth"),
        use_gpu=use_gpu,
    )
    _extractors_cache["iris"] = IrisExtractor(
        model_path=os.path.join(PROJECT_ROOT, "weights/iris/ResNet100_154000.pt"),
        use_gpu=use_gpu,
    )
    return _extractors_cache


def extract_embedding(trait, image_path, use_gpu=False):
    extractors = get_extractors(use_gpu)
    extractor  = extractors[trait]
    if trait == "iris":
        result = extractor.extract_features(image_path, backend="resnet100")
    else:
        result = extractor.extract_features(image_path)
    if result is None:
        raise RuntimeError(f"Feature extraction returned None for {trait}: {image_path}")
    return np.array(result, dtype=np.float32).flatten()


# ===========================================================================
# 4. STAGE 1 -- TRAIT-LEVEL 1:N BIOHASH SEARCH
# ===========================================================================

def search_trait_1n(probe_bh, trait, db_embeddings, top_k=5):
    scores = []
    for pid, traits in db_embeddings.items():
        if trait not in traits:
            continue
        sim = cosine_similarity(probe_bh, traits[trait])
        scores.append((pid, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def stage1_identify(probe_biohashes, db_embeddings, enroll_id, thresholds=None):
    """
    thresholds : dict with per-trait thresholds, e.g.
                 {"face": 0.30, "finger": 0.30, "iris": 0.20}
                 Defaults to DEFAULT_THR if not provided.
    """
    if thresholds is None:
        thresholds = DEFAULT_THR

    trait_results = {}
    vote_scores   = {}

    for trait, probe_bh in probe_biohashes.items():
        thr = thresholds.get(trait, DEFAULT_THR.get(trait, 0.30))
        top_matches = search_trait_1n(probe_bh, trait, db_embeddings, top_k=5)
        best_sim    = top_matches[0][1] if top_matches else 0.0
        best_id     = top_matches[0][0] if top_matches else None
        passed      = best_sim >= thr          # did this trait clear its own gate?
        trait_results[trait] = {
            "top_matches": top_matches,
            "best_sim"   : best_sim,
            "best_id"    : best_id,
            "threshold"  : thr,
            "passed"     : passed,
        }
        # Only traits that pass their own threshold contribute votes
        if passed:
            for pid, sim in top_matches:
                vote_scores[pid] = vote_scores.get(pid, 0.0) + max(sim, 0.0)

    if enroll_id:
        candidate_id = enroll_id
    else:
        if not vote_scores:
            return None, trait_results
        candidate_id = max(vote_scores, key=lambda p: vote_scores[p])

    # Verify at least one provided trait clears its threshold for the candidate
    best_any = max(
        (cosine_similarity(probe_bh, db_embeddings.get(candidate_id, {}).get(trait, np.zeros(HASH_DIM)))
         for trait, probe_bh in probe_biohashes.items()),
        default=0.0,
    )
    min_thr = min(thresholds.get(t, DEFAULT_THR.get(t, 0.30)) for t in probe_biohashes)
    if best_any < min_thr:
        return None, trait_results

    return candidate_id, trait_results


# ===========================================================================
# 5. STAGE 2 -- FUSED VERIFICATION
# ===========================================================================

def stage2_verify(probe_fused, candidate_id, all_fused_db, fused_threshold=None):
    if fused_threshold is None:
        fused_threshold = DEFAULT_THR["fused"]

    top_matches = []
    for pid, enrolled_fused in all_fused_db.items():
        sim = cosine_similarity(probe_fused, enrolled_fused)
        top_matches.append((pid, sim))
    top_matches.sort(key=lambda x: x[1], reverse=True)

    enrolled_fused = all_fused_db.get(candidate_id)
    id_sim = cosine_similarity(probe_fused, enrolled_fused) if enrolled_fused is not None else 0.0

    result_id = candidate_id if id_sim >= fused_threshold else None
    return result_id, id_sim, top_matches[:5], fused_threshold


# ===========================================================================
# 6. DISPLAY HELPERS
# ===========================================================================

def _bar(sim, width=30):
    filled = int(max(0.0, min(1.0, (sim + 1) / 2)) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_header():
    print()
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"{BOLD}{CYAN}   MULTIMODAL BIOMETRIC AUTHENTICATION SYSTEM{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"   Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Database  : biometric_final.db  |  Dataset: {DATASET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print()


def print_stage1(probe_traits, trait_results, candidate_id):
    print(f"{BOLD}-- STAGE 1 : Trait-Level BioHash Identification ----------{RESET}")
    print(f"   Probe traits supplied: {', '.join(probe_traits)}")
    print()
    for trait, res in trait_results.items():
        thr    = res.get("threshold", DEFAULT_THR.get(trait, 0.30))
        passed = res.get("passed", res["best_sim"] >= thr)
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"   {BOLD}{trait.upper()}{RESET}  "
              f"best={res['best_sim']:.4f} / thr={thr:.4f}  [{status}]  "
              f"-> {res['best_id']}  {_bar(res['best_sim'])}")
        for i, (pid, sim) in enumerate(res["top_matches"][:3], 1):
            marker = "*" if pid == candidate_id else " "
            print(f"      {marker} #{i:2d}  {pid}  sim={sim:.4f}")
    print()
    if candidate_id:
        print(f"   {GREEN}{BOLD}Candidate identified: {candidate_id}{RESET}")
    else:
        print(f"   {RED}No candidate matched above any trait threshold.{RESET}")
    print()


def print_stage2(candidate_id, recovered_traits, id_sim, top_matches, result_id, fused_threshold=None):
    if fused_threshold is None:
        fused_threshold = DEFAULT_THR["fused"]
    print(f"{BOLD}-- STAGE 2 : Fused Template Verification -----------------{RESET}")
    if recovered_traits:
        print(f"   {YELLOW}Recovered missing traits from DB: {', '.join(recovered_traits)}{RESET}")
    print(f"   CBP fusion: face + iris -> intermediate -> + finger")
    print()
    print(f"   1:N Fused Ranking (top 5):")
    for i, (pid, sim) in enumerate(top_matches, 1):
        marker = f"{GREEN}*{RESET}" if pid == candidate_id else " "
        print(f"      {marker} #{i:2d}  {pid}  sim={sim:.4f}  {_bar(sim)}")
    print()
    fused_status = f"{GREEN}PASS{RESET}" if result_id else f"{RED}FAIL{RESET}"
    print(f"   1:1 Fused Similarity [{candidate_id}]: {BOLD}{id_sim:.4f}{RESET}  "
          f"{_bar(id_sim)}  (threshold={fused_threshold:.4f})  [{fused_status}]")
    print()
    if result_id:
        print(f"{BOLD}{GREEN}{'=' * 65}{RESET}")
        print(f"{BOLD}{GREEN}   ACCESS GRANTED  --  Identity: {result_id}{RESET}")
        print(f"{BOLD}{GREEN}{'=' * 65}{RESET}")
    else:
        print(f"{BOLD}{RED}{'=' * 65}{RESET}")
        print(f"{BOLD}{RED}   ACCESS DENIED   --  Fused similarity below threshold{RESET}")
        print(f"{BOLD}{RED}{'=' * 65}{RESET}")
    print()


# ===========================================================================
# 7. MAIN AUTHENTICATION FUNCTION
# ===========================================================================

def authenticate(
    face_path=None,
    finger_path=None,
    iris_path=None,
    enroll_id=None,
    db_path=DEFAULT_DB,
    key_file=DEFAULT_KEY_FILE,
    use_gpu=False,
    verbose=True,
    thr_face=None,
    thr_finger=None,
    thr_iris=None,
    thr_fused=None,
):
    # Build per-trait threshold dict (fall back to defaults for any unset value)
    thresholds = {
        "face"  : thr_face   if thr_face   is not None else DEFAULT_THR["face"],
        "finger": thr_finger if thr_finger is not None else DEFAULT_THR["finger"],
        "iris"  : thr_iris   if thr_iris   is not None else DEFAULT_THR["iris"],
    }
    fused_threshold = thr_fused if thr_fused is not None else DEFAULT_THR["fused"]
    probe_inputs = {"face": face_path, "finger": finger_path, "iris": iris_path}
    provided_traits = {t: p for t, p in probe_inputs.items() if p}
    if not provided_traits:
        raise ValueError("Provide at least one probe trait image (--face / --finger / --iris).")

    if verbose:
        print_header()

    db           = BiometricDB(db_path)
    key_data     = load_biohash_keys(key_file)
    hashers      = build_biohashers(key_data)
    all_bh_db    = db.load_all_biohash_embeddings()
    all_fused_db = db.load_all_fused_embeddings()
    fi_fuser, ff_fuser = build_cbp_modules()

    if verbose:
        print(f"{CYAN}[INFO]{RESET} Enrolled persons: {len(all_bh_db)}  |  "
              f"Fused templates: {len(all_fused_db)}")
        print()

    # Extract and BioHash probe traits
    probe_biohashes = {}
    for trait, img_path in provided_traits.items():
        if verbose:
            print(f"{CYAN}[INFO]{RESET} Extracting {trait} from: {img_path}")
        raw_emb = extract_embedding(trait, img_path, use_gpu)
        probe_biohashes[trait] = hashers[trait].generate_biohash(raw_emb)

    # Stage 1: Trait-level identification
    candidate_id, trait_results = stage1_identify(probe_biohashes, all_bh_db, enroll_id, thresholds)

    if verbose:
        print_stage1(list(provided_traits.keys()), trait_results, candidate_id)

    if candidate_id is None:
        if verbose:
            print(f"{BOLD}{RED}{'=' * 65}{RESET}")
            print(f"{BOLD}{RED}   ACCESS DENIED   --  No candidate found in Stage 1{RESET}")
            print(f"{BOLD}{RED}{'=' * 65}{RESET}\n")
        db.close()
        return {"decision": "DENIED", "stage1_candidate": None, "trait_results": trait_results}

    # Recover missing traits from enrolled DB
    recovered_traits = []
    full_biohashes   = dict(probe_biohashes)

    for trait in TRAITS:
        if trait not in full_biohashes:
            enrolled_bh = db.load_biohash_embedding(candidate_id, trait)
            if enrolled_bh is not None:
                full_biohashes[trait] = enrolled_bh
                recovered_traits.append(trait)
            elif verbose:
                print(f"{YELLOW}[WARN]{RESET} Could not recover '{trait}' for {candidate_id}")

    if not all(t in full_biohashes for t in TRAITS):
        if verbose:
            print(f"{RED}[ERROR]{RESET} Cannot fuse: missing traits even after DB recovery.")
        db.close()
        return {"decision": "DENIED", "stage1_candidate": candidate_id, "reason": "Missing modality"}

    probe_fused = cbp_fuse_three(
        face_bh=full_biohashes["face"],
        iris_bh=full_biohashes["iris"],
        finger_bh=full_biohashes["finger"],
        fi_fuser=fi_fuser,
        ff_fuser=ff_fuser,
    )

    # Stage 2: Fused verification
    result_id, id_sim, fused_top5, _fthr = stage2_verify(probe_fused, candidate_id, all_fused_db, fused_threshold)

    if verbose:
        print_stage2(candidate_id, recovered_traits, id_sim, fused_top5, result_id, fused_threshold)

    db.close()

    return {
        "decision"        : "GRANTED" if result_id else "DENIED",
        "identity"        : result_id,
        "stage1_candidate": candidate_id,
        "fused_similarity": round(id_sim, 6),
        "fused_top5"      : [(pid, round(sim, 6)) for pid, sim in fused_top5],
        "trait_results"   : {
            t: {"best_id": r["best_id"], "best_sim": round(r["best_sim"], 6)}
            for t, r in trait_results.items()
        },
        "recovered_traits": recovered_traits,
    }


# ===========================================================================
# 8. CLI ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
            Multimodal Biometric Authentication
            ------------------------------------
            Provide 1-3 trait images. Missing traits are recovered
            from the database for the matched/specified identity.

            Examples:
              python authenticate.py --face probe_face.jpg --enroll-id Person_001
              python authenticate.py --face f.jpg --finger fp.jpg --iris i.jpg
              python authenticate.py --finger fp.jpg --iris i.jpg
        """)
    )
    parser.add_argument("--face",       type=str,   default=None)
    parser.add_argument("--thr-face",   type=float, default=None,
                        help=f"Face biohash similarity threshold (default {DEFAULT_THR['face']})")
    parser.add_argument("--thr-finger", type=float, default=None,
                        help=f"Finger biohash similarity threshold (default {DEFAULT_THR['finger']})")
    parser.add_argument("--thr-iris",   type=float, default=None,
                        help=f"Iris biohash similarity threshold (default {DEFAULT_THR['iris']})")
    parser.add_argument("--thr-fused",  type=float, default=None,
                        help=f"Fused template similarity threshold (default {DEFAULT_THR['fused']})")
    parser.add_argument("--finger",     type=str,   default=None)
    parser.add_argument("--iris",       type=str,   default=None)
    parser.add_argument("--enroll-id", type=str, default=None,
                        help="Enrolled person_id for 1:1 verification (e.g. Person_001)")
    parser.add_argument("--db",        type=str, default=DEFAULT_DB)
    parser.add_argument("--key-file",  type=str, default=DEFAULT_KEY_FILE)
    parser.add_argument("--gpu",       action="store_true")
    parser.add_argument("--quiet",     action="store_true")

    args = parser.parse_args()

    result = authenticate(
        face_path=args.face,
        finger_path=args.finger,
        iris_path=args.iris,
        enroll_id=args.enroll_id,
        db_path=args.db,
        key_file=args.key_file,
        use_gpu=args.gpu,
        verbose=not args.quiet,
        thr_face=args.thr_face,
        thr_finger=args.thr_finger,
        thr_iris=args.thr_iris,
        thr_fused=args.thr_fused,
    )

    import json as _json
    print(f"\n{BOLD}-- JSON Result -------------------------------------------{RESET}")
    print(_json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
