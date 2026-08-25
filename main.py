"""
main.py — Multimodal biometric enrollment pipeline.

Responsibility: load extractors once, walk a dataset directory,
generate embeddings for face + fingerprint + iris per person,
and persist the template database to disk.

Verification and identification are available as methods on
MultiModalBiometricPipeline for use by other scripts or services
that import this module — they are not part of this file's CLI.

Usage
-----
    python main.py --data-dir /path/to/dataset
    python main.py --data-dir /path/to/dataset --db enrolled.json --gpu
"""

import os
import json
import logging
import sqlite3
import tempfile
import argparse
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import numpy as np

from extractors.face_extractor import FaceExtractor
from extractors.finger_extractor import FingerprintExtractor
from fingerprint import DeepPrintFingerprintExtractor
from iris.extractor.iris_extractor import IrisExtractor

logger = logging.getLogger(__name__)


# Default image filenames expected inside each person's sub-directory.
# Override by passing `image_filenames=` to enroll_from_directory().
DEFAULT_IMAGE_FILENAMES = {
    "face":   "face.jpg",
    "finger": "fingerprint_right_thumb.jpg",
    "iris":   "iris_right.jpg",
}


class MultiModalBiometricPipeline:
    """
    Orchestrates all three extractors + score-level fusion.

    Extractors are loaded once in __init__ (expensive) and reused
    across every extraction call (cheap). Never re-instantiate them
    inside a loop.
    """

    def __init__(
        self,
        face_model_path: str = "weights/face/w600k_r50.onnx",
        face_detector_path: str = "weights/face/det_10g.onnx",
        finger_model_path: Optional[str] = None,
        finger_backend: str = "deepprint",
        iris_model_path: str = "weights/iris/ResNet100_154000.pt",
        iris_seg_model_path: str = "weights/iris/iris_semseg_upp_scse_mobilenetv2.onnx",
        iris_circlenet_path: str = "weights/iris/resnet18-1543-0.047488-maskIoU-0.934494.pth",
        iris_seg_backend: str = "circlenet",
        use_gpu: bool = False,
    ):
        logger.info("Loading face extractor...")
        self.face_extractor = FaceExtractor(
            model_path=face_model_path,
            detector_path=face_detector_path,
        )

        logger.info(f"Loading fingerprint extractor (backend: {finger_backend})...")
        if finger_backend == "deepprint":
            actual_finger_path = finger_model_path or "weights/finger/best_model.pyt"
            self.finger_extractor = DeepPrintFingerprintExtractor(
                model_path=actual_finger_path,
                model_type="deepprint_texminu",
                use_gpu=use_gpu,
            )
        else:
            actual_finger_path = finger_model_path or "weights/finger/finger_extractor_best.pth"
            self.finger_extractor = FingerprintExtractor(
                model_path=actual_finger_path,
                use_gpu=use_gpu,
            )

        logger.info(f"Loading iris extractor (seg_backend: {iris_seg_backend} + IResNet-100)...")
        self.iris_extractor = IrisExtractor(
            model_path=iris_model_path,
            seg_model_path=iris_seg_model_path,
            circlenet_model_path=iris_circlenet_path,
            seg_backend=iris_seg_backend,
            use_gpu=use_gpu,
        )

        # person_id -> {"face": np.ndarray, "finger": np.ndarray, "iris": np.ndarray}
        self.database: dict = {}
        # person_id -> list of live sets: [{"live_index": int, "face": np.ndarray, "finger": np.ndarray, "iris": np.ndarray}, ...]
        self.live_database: dict = {}

        # Optional SQLite connection (set via init_sqlite_db)
        self._sqlite_conn: sqlite3.Connection = None

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_all(self, face_img, finger_img, iris_img):
        """Run all three extractors and return (face_emb, finger_emb, iris_emb)."""
        face_emb   = self.face_extractor.extract_features(face_img, check_quality=True)
        finger_emb = self.finger_extractor.extract_features(finger_img)
        iris_emb   = self.iris_extractor.extract_features(iris_img)
        return face_emb, finger_emb, iris_emb

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------
    def enroll_person(self, person_id: str, face_img, finger_img, iris_img):
        """Extract and store one person's template."""
        face_emb, finger_emb, iris_emb = self.extract_all(face_img, finger_img, iris_img)
        self.database[person_id] = {
            "face":   face_emb,
            "finger": finger_emb,
            "iris":   iris_emb,
        }
        logger.info(f"Enrolled: {person_id}")

    def enroll_from_directory(self, root_dir: str, image_filenames: dict = None):
        """
        Batch-enroll every person sub-folder found under root_dir.

        Supports two layouts:

        1. Single-image (original):
              person_001/
                face.jpg
                fingerprint_right_thumb.jpg
                iris_right.jpg

        2. Multi-image subfolders (setA_std):
              person_001/
                face/    face_01.jpg, face_02.jpg, ...
                finger/  finger_01.jpg, finger_02.jpg, ...
                iris/    iris_01.jpg, iris_02.jpg, ...

           All images in each subfolder are embedded and averaged into
           a single L2-normalized template stored in the database.

        Parameters
        ----------
        root_dir : str
            Dataset root directory.
        image_filenames : dict, optional
            Override default filenames for single-image layout.
            Keys: ``'face'``, ``'finger'``, ``'iris'``.

        Returns
        -------
        succeeded : list[str]
        failed    : list[str]
        """
        if image_filenames is None:
            image_filenames = DEFAULT_IMAGE_FILENAMES

        valid_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

        # Subfolder names used in the multi-image layout
        MODALITY_SUBDIRS = {"face": "face", "finger": "finger", "iris": "iris"}

        def get_images_from_subdir(person_dir, subdir_name):
            """Return sorted list of image paths from a modality subfolder."""
            subdir = os.path.join(person_dir, subdir_name)
            if not os.path.isdir(subdir):
                return []
            return sorted([
                os.path.join(subdir, f)
                for f in os.listdir(subdir)
                if os.path.splitext(f)[1].lower() in valid_ext
            ])

        def average_embeddings(emb_list):
            """
            Removes the farthest outlier before averaging to prevent poisoning.
            """
            if len(emb_list) == 1:
                return (emb_list[0] / max(np.linalg.norm(emb_list[0]), 1e-12)).astype(np.float32)
            
            stacked = np.stack(emb_list, axis=0)  # (N, D)
            
            # Step 1: Calculate the initial average
            mean_init = stacked.mean(axis=0)
            mean_init = mean_init / max(np.linalg.norm(mean_init), 1e-12)
            
            # Step 2: Find the image that is MOST dissimilar (farthest cosine distance) to this average
            similarities = np.dot(stacked, mean_init)  # Cosine similarity (since both are L2-normed)
            worst_idx = np.argmin(similarities)        # Index of the biggest outlier
            
            # Step 3: Remove the outlier and re-average
            pruned_list = [emb for i, emb in enumerate(emb_list) if i != worst_idx]
            stacked_pruned = np.stack(pruned_list, axis=0)
            final_mean = stacked_pruned.mean(axis=0)
            final_mean = final_mean / max(np.linalg.norm(final_mean), 1e-12)
            
            return final_mean.astype(np.float32)

        person_ids = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

        succeeded, failed = [], []\
        
        for person_id in person_ids:
            person_dir = os.path.join(root_dir, person_id)

            # Detect layout: multi-image subfolders take priority
            face_imgs   = get_images_from_subdir(person_dir, MODALITY_SUBDIRS["face"])
            finger_imgs = get_images_from_subdir(person_dir, MODALITY_SUBDIRS["finger"])
            iris_imgs   = get_images_from_subdir(person_dir, MODALITY_SUBDIRS["iris"])

            use_multi = bool(face_imgs or finger_imgs or iris_imgs)

            if not use_multi:
                # Fall back to single-image layout
                face_path   = os.path.join(person_dir, image_filenames["face"])
                finger_path = os.path.join(person_dir, image_filenames["finger"])
                iris_path   = os.path.join(person_dir, image_filenames["iris"])

                missing = [
                    name for name, path in [
                        ("face",   face_path),
                        ("finger", finger_path),
                        ("iris",   iris_path),
                    ]
                    if not os.path.exists(path)
                ]
                if missing:
                    logger.warning(f"Skipping {person_id}: missing modality files: {missing}")
                    failed.append(person_id)
                    continue

                try:
                    self.enroll_person(person_id, face_path, finger_path, iris_path)
                    succeeded.append(person_id)
                except Exception as e:
                    logger.error(f"Failed to enroll {person_id}: {e}")
                    failed.append(person_id)
                continue

            # --- Multi-image enrollment & Live set creation ---
            try:
                # Partition image lists: first 3 images for enrollment averaging, remaining for live set
                enroll_face_paths   = face_imgs[:3]
                leftover_face_paths = face_imgs[3:]

                enroll_finger_paths   = finger_imgs[:3]
                leftover_finger_paths = finger_imgs[3:]

                enroll_iris_paths   = iris_imgs[:3]
                leftover_iris_paths = iris_imgs[3:]

                # 1. Embed enrollment images (up to 3 per trait)
                enroll_face_embs = []
                for img_path in enroll_face_paths:
                    try:
                        emb = self.face_extractor.extract_features(img_path, check_quality=False)
                        enroll_face_embs.append(emb)
                    except Exception as e:
                        logger.warning(f"  [face enrollment] {person_id} {os.path.basename(img_path)}: {e}")

                enroll_finger_embs = []
                for img_path in enroll_finger_paths:
                    try:
                        emb = self.finger_extractor.extract_features(img_path)
                        enroll_finger_embs.append(emb)
                    except Exception as e:
                        logger.warning(f"  [finger enrollment] {person_id} {os.path.basename(img_path)}: {e}")

                enroll_iris_embs = []
                for img_path in enroll_iris_paths:
                    try:
                        emb = self.iris_extractor.extract_features(img_path)
                        enroll_iris_embs.append(emb)
                    except Exception as e:
                        logger.warning(f"  [iris enrollment] {person_id} {os.path.basename(img_path)}: {e}")

                if not (enroll_face_embs and enroll_finger_embs and enroll_iris_embs):
                    logger.warning(
                        f"Skipping {person_id}: insufficient enrollment embeddings "
                        f"(face={len(enroll_face_embs)}, finger={len(enroll_finger_embs)}, iris={len(enroll_iris_embs)})"
                    )
                    failed.append(person_id)
                    continue

                # Average 3 enrollment embeddings -> single L2-normalized template per modality
                avg = {
                    "face":   average_embeddings(enroll_face_embs),
                    "finger": average_embeddings(enroll_finger_embs),
                    "iris":   average_embeddings(enroll_iris_embs),
                }
                self.database[person_id] = avg

                # 2. Embed leftover images for live set creation
                leftover_face_embs = []
                for img_path in leftover_face_paths:
                    try:
                        emb = self.face_extractor.extract_features(img_path, check_quality=False)
                        leftover_face_embs.append(emb)
                    except Exception as e:
                        logger.warning(f"  [face live] {person_id} {os.path.basename(img_path)}: {e}")

                leftover_finger_embs = []
                for img_path in leftover_finger_paths:
                    try:
                        emb = self.finger_extractor.extract_features(img_path)
                        leftover_finger_embs.append(emb)
                    except Exception as e:
                        logger.warning(f"  [finger live] {person_id} {os.path.basename(img_path)}: {e}")

                leftover_iris_embs = []
                gal_iris_emb = avg["iris"]
                for img_path in leftover_iris_paths:
                    try:
                        emb = self.iris_extractor.extract_features_aligned(img_path, gallery_embedding=gal_iris_emb)
                        leftover_iris_embs.append(emb)
                    except Exception as e:
                        logger.warning(f"  [iris live] {person_id} {os.path.basename(img_path)}: {e}")


                # 3. Create live set for leftover images (with cycling if a trait's images run out)
                # Number of live set iterations = max count of leftover embeddings across traits
                num_live = max(len(leftover_face_embs), len(leftover_finger_embs), len(leftover_iris_embs))
                if num_live == 0:
                    num_live = 1

                live_sets = []
                for k in range(num_live):
                    live_idx = k + 1
                    # Select per-image embeddings, cycling if trait images run out
                    f_emb  = leftover_face_embs[k % len(leftover_face_embs)] if leftover_face_embs else enroll_face_embs[k % len(enroll_face_embs)]
                    fg_emb = leftover_finger_embs[k % len(leftover_finger_embs)] if leftover_finger_embs else enroll_finger_embs[k % len(enroll_finger_embs)]
                    ir_emb = leftover_iris_embs[k % len(leftover_iris_embs)] if leftover_iris_embs else enroll_iris_embs[k % len(enroll_iris_embs)]

                    live_sets.append({
                        "live_index": live_idx,
                        "face":   f_emb,
                        "finger": fg_emb,
                        "iris":   ir_emb,
                    })

                self.live_database[person_id] = live_sets

                # All individual embeddings dictionary for backwards compatibility
                all_embs = {
                    "face":   enroll_face_embs + leftover_face_embs,
                    "finger": enroll_finger_embs + leftover_finger_embs,
                    "iris":   enroll_iris_embs + leftover_iris_embs,
                }

                # Persist to SQLite if a connection is open
                if self._sqlite_conn is not None:
                    self.save_sqlite_db(
                        person_id,
                        avg,
                        all_embeddings=all_embs,
                        live_sets=live_sets,
                    )

                logger.info(
                    f"Enrolled {person_id}: "
                    f"enroll_avg(face={len(enroll_face_embs)}, finger={len(enroll_finger_embs)}, iris={len(enroll_iris_embs)}), "
                    f"live_sets={len(live_sets)} (leftovers: face={len(leftover_face_embs)}, finger={len(leftover_finger_embs)}, iris={len(leftover_iris_embs)})"
                )
                succeeded.append(person_id)

            except Exception as e:
                logger.error(f"Failed to enroll {person_id}: {e}")
                failed.append(person_id)

        logger.info(
            f"Enrollment complete: {len(succeeded)} succeeded, {len(failed)} failed."
        )
        if failed:
            logger.info(f"Failed person IDs: {failed}")
        return succeeded, failed


    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------
    def save_database(self, path: str, live_path: str = None):
        """
        Atomically serialize enrolled embeddings to JSON (and optionally live sets to live_path).
        Uses tempfile + os.replace so a crash never corrupts the output.
        """
        serializable = {
            pid: {k: v.tolist() for k, v in traits.items()}
            for pid, traits in self.database.items()
        }
        dir_name = os.path.dirname(os.path.abspath(path))
        with tempfile.NamedTemporaryFile("w", dir=dir_name, suffix=".tmp", delete=False) as tmp:
            tmp_path = tmp.name
            json.dump(serializable, tmp, indent=2)
        os.replace(tmp_path, path)
        logger.info(f"Saved {len(self.database)} enrolled templates to {path}")

        if live_path and self.live_database:
            live_serializable = {
                pid: [
                    {
                        "live_index": sample["live_index"],
                        **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sample.items() if k != "live_index"}
                    }
                    for sample in samples
                ]
                for pid, samples in self.live_database.items()
            }
            live_dir = os.path.dirname(os.path.abspath(live_path))
            with tempfile.NamedTemporaryFile("w", dir=live_dir, suffix=".tmp", delete=False) as tmp:
                l_tmp_path = tmp.name
                json.dump(live_serializable, tmp, indent=2)
            os.replace(l_tmp_path, live_path)
            logger.info(f"Saved {len(self.live_database)} live database entries to {live_path}")

    def load_database(self, path: str):
        """Load a previously saved enrollment database from JSON."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Database file not found: {path}")
        with open(path) as f:
            raw = json.load(f)
        self.database = {
            pid: {k: np.array(v, dtype=np.float32) for k, v in traits.items()}
            for pid, traits in raw.items()
        }
        logger.info(f"Loaded {len(self.database)} enrolled templates from {path}")

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------
    def init_sqlite_db(self, db_path: str, dataset_tag: str = "default"):
        """
        Open (or create) a SQLite database and set up the schema.
        Call this BEFORE enroll_from_directory() to enable automatic
        SQLite writes during enrollment.

        Schema
        ------
        persons          — one row per (person, dataset)
        templates        — averaged L2-normalized embedding per person per trait
        image_embeddings — one row per individual image embedding (all modalities)

        Embeddings are stored as raw float32 BLOBs (tobytes / frombuffer) —
        zero precision loss, exact byte-for-byte round-trip.
        """
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._sqlite_conn = sqlite3.connect(db_path)
        self._sqlite_dataset = dataset_tag
        cur = self._sqlite_conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id  TEXT NOT NULL,
                dataset    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (person_id, dataset)
            );
            CREATE TABLE IF NOT EXISTS EMBEDDINGS (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  TEXT    NOT NULL,
                dataset    TEXT    NOT NULL,
                trait      TEXT    NOT NULL,
                embedding  BLOB    NOT NULL,
                dim        INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS image_embeddings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id    TEXT    NOT NULL,
                dataset      TEXT    NOT NULL,
                trait        TEXT    NOT NULL,
                image_index  INTEGER NOT NULL,
                embedding    BLOB    NOT NULL,
                dim          INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_embeddings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id    TEXT    NOT NULL,
                dataset      TEXT    NOT NULL,
                live_index   INTEGER NOT NULL,
                trait        TEXT    NOT NULL,
                embedding    BLOB    NOT NULL,
                dim          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tmpl  ON EMBEDDINGS      (person_id, dataset, trait);
            CREATE INDEX IF NOT EXISTS idx_img   ON image_embeddings(person_id, dataset, trait, image_index);
            CREATE INDEX IF NOT EXISTS idx_live  ON live_embeddings (person_id, dataset, live_index, trait);
        """)
        self._sqlite_conn.commit()
        logger.info(f"SQLite DB initialized: {db_path}  (dataset='{dataset_tag}')")

    def save_sqlite_db(
        self,
        person_id: str,
        avg_embeddings: dict,
        all_embeddings: dict = None,
        live_sets: list = None,
    ):
        """
        Write one person's data to SQLite.

        Parameters
        ----------
        person_id     : e.g. 'Person_001'
        avg_embeddings: {trait: np.ndarray}  — averaged template per modality
        all_embeddings: {trait: [np.ndarray, ...]}  — every image embedding;
                        pass None to only store the averaged template.
        live_sets     : list[dict] — per-image live set triples with cycling
        """
        if self._sqlite_conn is None:
            raise RuntimeError("Call init_sqlite_db() before save_sqlite_db().")

        now = datetime.now(timezone.utc).isoformat()
        dataset = self._sqlite_dataset
        cur = self._sqlite_conn.cursor()

        # persons row
        cur.execute(
            "INSERT OR IGNORE INTO persons (person_id, dataset, created_at) VALUES (?,?,?)",
            (person_id, dataset, now),
        )

        for trait, avg_emb in avg_embeddings.items():
            arr = np.array(avg_emb, dtype=np.float32)
            # averaged template
            cur.execute(
                "INSERT INTO EMBEDDINGS (person_id, dataset, trait, embedding, dim) VALUES (?,?,?,?,?)",
                (person_id, dataset, trait, arr.tobytes(), arr.shape[0]),
            )

            # individual image embeddings
            if all_embeddings and trait in all_embeddings:
                for idx, emb in enumerate(all_embeddings[trait]):
                    img_arr = np.array(emb, dtype=np.float32)
                    cur.execute(
                        "INSERT INTO image_embeddings "
                        "(person_id, dataset, trait, image_index, embedding, dim) VALUES (?,?,?,?,?,?)",
                        (person_id, dataset, trait, idx, img_arr.tobytes(), img_arr.shape[0]),
                    )

        if live_sets:
            for live_item in live_sets:
                l_idx = live_item["live_index"]
                for trait in ("face", "finger", "iris"):
                    if trait in live_item:
                        l_arr = np.array(live_item[trait], dtype=np.float32)
                        cur.execute(
                            "INSERT INTO live_embeddings "
                            "(person_id, dataset, live_index, trait, embedding, dim) VALUES (?,?,?,?,?,?)",
                            (person_id, dataset, l_idx, trait, l_arr.tobytes(), l_arr.shape[0]),
                        )

        self._sqlite_conn.commit()
        logger.debug(f"SQLite: saved {person_id}")

    def close_sqlite_db(self):
        """Flush and close the SQLite connection."""
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
            self._sqlite_conn = None
            logger.info("SQLite DB connection closed.")

    # ------------------------------------------------------------------
    # Fusion helpers  (used by verify / identify)
    # ------------------------------------------------------------------
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    @staticmethod
    def fuse_scores(
        s_face: float, s_finger: float, s_iris: float,
        weights: tuple = (0.3, 0.35, 0.35),
    ) -> float:
        """
        Weighted-sum score fusion.
        Default weights favour finger/iris (typically more discriminative than face).
        Replace with EER-derived weights after validating on your dataset:
            w_i = (1/EER_i) / sum(1/EER_j)
        """
        w_face, w_finger, w_iris = weights
        return w_face * s_face + w_finger * s_finger + w_iris * s_iris

    # ------------------------------------------------------------------
    # Verification  (1:1)
    # ------------------------------------------------------------------
    def verify(
        self,
        person_id: str, face_img, finger_img, iris_img,
        threshold: float = 0.55,
        weights: tuple = (0.3, 0.35, 0.35),
    ) -> dict:
        """1:1 — compare a probe against one claimed enrolled identity."""
        if person_id not in self.database:
            raise KeyError(f"No enrolled template for person_id={person_id!r}")

        enrolled = self.database[person_id]
        probe_face, probe_finger, probe_iris = self.extract_all(face_img, finger_img, iris_img)

        s_face   = self.cosine_similarity(probe_face,   enrolled["face"])
        s_finger = self.cosine_similarity(probe_finger, enrolled["finger"])
        s_iris   = self.cosine_similarity(probe_iris,   enrolled["iris"])
        fused    = self.fuse_scores(s_face, s_finger, s_iris, weights)

        return {
            "person_id":   person_id,
            "scores":      {"face": round(s_face, 6), "finger": round(s_finger, 6), "iris": round(s_iris, 6)},
            "fused_score": round(fused, 6),
            "threshold":   threshold,
            "decision":    "ACCEPT" if fused >= threshold else "REJECT",
        }

    # ------------------------------------------------------------------
    # Identification  (1:N)
    # ------------------------------------------------------------------
    def identify(
        self,
        face_img, finger_img, iris_img,
        threshold: float = 0.55,
        weights: tuple = (0.3, 0.35, 0.35),
    ) -> dict:
        """1:N — find the best matching identity across all enrolled people."""
        if not self.database:
            raise RuntimeError("Database is empty. Enroll at least one person first.")

        probe_face, probe_finger, probe_iris = self.extract_all(face_img, finger_img, iris_img)

        best_person, best_score = None, -1.0
        for person_id, enrolled in self.database.items():
            fused = self.fuse_scores(
                self.cosine_similarity(probe_face,   enrolled["face"]),
                self.cosine_similarity(probe_finger, enrolled["finger"]),
                self.cosine_similarity(probe_iris,   enrolled["iris"]),
                weights,
            )
            if fused > best_score:
                best_score, best_person = fused, person_id

        matched = best_score >= threshold
        return {
            "person_id":   best_person if matched else None,
            "fused_score": round(best_score, 6),
            "threshold":   threshold,
            "decision":    "MATCH" if matched else "NO_MATCH",
        }


# ----------------------------------------------------------------------
# Entry point — enroll a dataset, save the database
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Enroll a biometric dataset and save the embedding database."
    )
    parser.add_argument("--data-dir",  required=True, help="Dataset root directory.")
    parser.add_argument("--db",        default="enrolled_templates.json", help="Output JSON database path.")
    parser.add_argument("--sqlite",    default=None, help="Optional SQLite DB path (e.g. database/biometric.db).")
    parser.add_argument("--dataset",   default="default", help="Dataset tag stored in SQLite (e.g. setA).")
    parser.add_argument("--face-model",    default="weights/face/w600k_r50.onnx")
    parser.add_argument("--face-detector", default="weights/face/det_10g.onnx")
    parser.add_argument("--finger-backend", choices=["deepprint", "resnet50"], default="deepprint", help="Fingerprint feature extractor backend.")
    parser.add_argument("--finger-model",  default="weights/finger/best_model.pyt", help="Fingerprint weights path (default: weights/finger/best_model.pyt for deepprint).")
    parser.add_argument("--iris-model",    default="weights/iris/ResNet100_154000.pt")
    parser.add_argument("--iris-seg-model", default="weights/iris/iris_semseg_upp_scse_mobilenetv2.onnx")
    parser.add_argument("--iris-circlenet-model", default="weights/iris/resnet18-1543-0.047488-maskIoU-0.934494.pth")
    parser.add_argument("--iris-seg", choices=["circlenet", "unet", "hough"], default="circlenet", help="Iris segmentation backend.")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU if available.")
    args = parser.parse_args()

    pipeline = MultiModalBiometricPipeline(
        face_model_path=args.face_model,
        face_detector_path=args.face_detector,
        finger_model_path=args.finger_model,
        finger_backend=args.finger_backend,
        iris_model_path=args.iris_model,
        iris_seg_model_path=args.iris_seg_model,
        iris_circlenet_path=args.iris_circlenet_model,
        iris_seg_backend=args.iris_seg,
        use_gpu=args.gpu,
    )

    if args.sqlite:
        pipeline.init_sqlite_db(args.sqlite, dataset_tag=args.dataset)

    pipeline.enroll_from_directory(args.data_dir)
    pipeline.save_database(args.db)

    if args.sqlite:
        pipeline.close_sqlite_db()
        logger.info(f"SQLite DB saved: {args.sqlite}")