# activating biofuse virtual environment

# this is for my local pc to activate a virtual environment created for this research project

cd..;cd..;cd coding;cd biometric_fusion_system;biofuse_venv/scripts/activate;cd..;cd..;cd biometrics_intern;cd biometrics_fusion

# First run: enrolls all images and caches embeddings

python eval.py --data-dir data/all_images/setA_std --db embeddings/setA_db.json --out results.json

# Subsequent runs: loads cached DB (skips re-enrollment, much faster)

python eval.py --data-dir data/all_images/setA_std --db embeddings/setA_db.json --out results.json

# new evaluation

python biometric_eval_matrix.py --embeddings embeddings/embeddings.json --out_dir results

python biometric_eval_matrix.py --db database/biometric.db --dataset setA --out_dir results/setA_eval

# fussed

python fused_eval_matrix.py --db database/biometric.db --dataset setA --out_dir results/fused_eval

# concat fusion & evaluation

python concat_fusion_db.py --db database/biometric.db --dataset setA

python fused_eval_matrix.py --db database/biometric.db --dataset setA --fusion_type concat_l2 --out_dir results/fused_eval

# compact bilinear fusion (CBP) & evaluation

python cbp_fusion_db.py --db database/biometric_final.db --dataset setA --output_dim 2048

python fused_eval_matrix.py --db database/biometric_final.db --dataset setA --fusion_type cbp --out_dir results/cbp_fused_eval

# traditional biohashing generation & protection

python biohashing.py --db database/biometric.db --dataset setA --hash_dim 512 --key_seed 2026 --keys_out database/biohash_keys.json

# biohashed single-modality evaluation

python biometric_eval_matrix.py --db database/biometric.db --dataset setA --template_type biohash --out_dir results/biohash_setA_eval

# biohashed fused cbp evaluation

python fused_eval_matrix.py --db database/biometric.db --dataset setA --fusion_type cbp --template_type biohash --out_dir results/biohash_cbp_eval

# main  run

python main.py --data-dir data/all_images/setA_std --db enrolled_templates.json --sqlite database/biometric.db --dataset setA

# clear tables from db

python clear_tables.py database/biometric.db biohash_templates

# 1:1 — provide face + finger, iris recovered from DB

python authenticate.py --face probe_face.jpg --finger probe_finger.jpg --enroll-id Person_001

# 1:N — provide all 3 traits, system identifies automatically

python authenticate.py --face f.jpg --finger fp.jpg --iris i.jpg

# Single trait probe with 1:1 verification

python authenticate.py --face probe_face.jpg --enroll-id Person_045

# With GPU

python authenticate.py --face f.jpg --iris i.jpg --gpu

# Silent mode (only JSON output)

python authenticate.py --face f.jpg --enroll-id Person_003 --quiet

# verification

python authenticate.py --face D:\biometrics_intern\biometrics_fusion\data\all_images\setA\Person_036\face\36-06.jpg  --finger 'D:\biometrics_intern\biometrics_fusion\data\all_images\setA\Person_036\fingerprint right thumb\36_6.tif'  --thr-face 0.2971 --thr-finger 0.5474 --thr-fused 0.0313

python authenticate.py --face D:\biometrics_intern\biometrics_fusion\data\all_images\setA\Person_018\face\18-05.jpg   --finger 'D:\biometrics_intern\biometrics_fusion\data\all_images\setA\Person_018\fingerprint right thumb\18_4.tif'  --thr-face 0.2971 --thr-finger 0.5474 --thr-fused 0.0313

python authenticate.py --iris 'D:\biometrics_intern\biometrics_fusion\data\all_images\setA_std\Person_066\iris\iris_06.jpg'  --finger 'D:\biometrics_intern\biometrics_fusion\data\all_images\setA_std\Person_066\finger\finger_04.jpg'  --thr-iris 0.1289 --thr-finger 0.5474 --thr-fused 0.0313
