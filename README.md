# YOLO-GunRecognizer-V0

- **THERE WAS A DATA LEAKAGE FOR V3 EXPERIMENT (HARDNEGS) - PROJECT IN PROGRESS**


`YOLO-GunRecognizer-v0`. Single-class ("gun") computer vision model trained to detect
firearms. The YOLO architecture was used to make the model deployable in real world cameras for live detection.

Key benchmark:
    1. UGR handgun benchmark - published benchmark deduped against training data

### TL;DR
A single-class YOLO model (class: "gun") designed for live-monitoring and real-surveillance frames. 
- Main problem: data distribution. While at first, hyperparameter-tunning was 
considered as the best first approach, such techniques barely improved the model. However, by performing data techniques such as downsampling, including hard-negatives, etc. The model improved by 3.65x (recall).

- **Shipped model: v3**: UGR mAP 0.856, OOD Recall 0.53, 1% False-Positives.
                         46-85 FPS on RTX 4060  

- **THERE WAS A DATA LEAKAGE FOR V3 EXPERIMENT (HARDNEGS) - PROJECT IN PROGRESS**

## Key Files

- `add_roboflow_data.py`
script to download selected roboflow datasets.
- `add_hard_negatives.py`
Deletion of hard-negative photo labels. creation of .yaml 
- `check_leakage.py`
Checks data leakage from new datasets. creates delete_roboflow_duplicates.sh
to automatically remove data leakage throughout the about-to-merge dataset.
- `deploy_model.py`
deploys computer vision model into roboflow.
- `eval_hard_neg.py` / `eval_slices.py` / `eval_ugr.py`
evaluation scripts to obtain results
- `experiment_%j.slurm`
slurm scripts used in HPC Cluster to submit jobs
- `make_train` / `make_downsample`
.yaml generators specific to each experiment
- `resplit_cctv_by_video.py` 
script used to randomly resplit cctv data by 1/3
- `train_&j.py` 
scripts corresponding to each experiment
(scripts submitted through hpc clusters using slurm files).



### Training data

~12,693 frames in dataset (train: 11,327 - val: 679 - test: 687).

- `v3_model`
(deployable) train/tested/eval on 10,380 frames
(train: 9,014, val: 679, test: 687). 
- `v3 ratio`
-> train/val/test -- 87% / 6.5% / 6.6%
- `UGR benchmark`
2,879 clean frames. 3000 evaluation frames separate from split dataset. (121 images were dropped due to overlapping training).

used downsample techniques and hard-negatives integration in order to improve,
recall and precision while minimizing overfitting.

**The original datasets carry their own license (typically non-commercial
for roboflow datasets) - verify before redistribution or commercial use.**

## Evaluation 
- PER-SLICE EVALUATION. Metrics are reported separately for catalog, staged CCTV 
and real-CCTV (youtube) instead of one average metric. 
- VIDEO-LEVEL-SPLITS. frames from the same resource video were stored in the same
usage folder (either train/split) to avoid data leakage in evaluation.
- PERCEPTUAL-HASH DEDUP (dHash). Every addition made to the dataset was hash-checked
against test set. Examples
    * pistol-csvic: 19 frames
    * cctv-gun-detector: 69 frames (from the same resource video)
    * UGR benchmark: 121 frames of the 3000 images overlapped with training data
- R@conf / P@conf: Suggested fixed-confidence for deployment. These values return 
recall and precision with suggested confidence.

test/youtube: real robbery CCTV (less/confusing image quality)
val/cctv: staged cctv (reads 0.99-1 for every model?)

## Experiments

### baseline (yolo26n, imgsz 640)
- Config      : YOLO26n (nano), 640px, standard augmentation.
- Hypothesis  : Establish a floor.
- Result      : test/youtube R 0.149 ; test/openimages mAP50 ~0.666.
              Best epoch 83, catalog mAP50-95 0.481.
- Conclusion     : Decent on catalog guns, improvement needed on real footage.

### yolo26s_960_aug  

- Config      : (yolo26s, imgsz 960, mixup 0.1, copy_paste 0.2, cos_lr)
- Hypothesis  : A bigger/better-tuned model will generalize better.
- Result      : test/youtube R 0.199 (+0.05) ; test/openimages mAP50 0.730 (+0.06).
- Conclusion  : Real-world recal (test/youtube) moved only by 5%. pointing more to
                a data problem rather than an architecture problem.

### yolo26s_960_aug_2  

- Config      : Added Gun-Movies-Database CCTV frames; fixed frame-leakage with a
              video-level train/val/test split.
- Hypothesis  : Surveillance data will fix deployable/real-world recall
- Result      : test/youtube R 0.450 (up from 0.199 — a 2.3x jump).
                val/cctv R 0.99 - exposed data leakage.
                val/openimages R 0.68 - neutral 
- Conclusion  : model was overfitting val/cctv while performing badly on real-world
                low quality images (even though there was a 2.3x jump)

### yolo26s_1280_diverse  (+ guns-r46kc "diverse" data, imgsz 1280)

- Config      : Added arads1/guns-r46kc dataset (robbery/movie frames); imgsz 1280;
                batch 16; scale 0.9
- Hypothesis  : More DIVERSE real-ish data + higher res pushes OOD further.
- Result      : test/youtube R 0.494 (only +0.04 over CCTV run).
              mAP50 0.551, mAP50-95 0.219, P 0.759.
              test/openimages mAP50 0.762.
- Conclusion  : 1280 imgsz improved results. Low recall & high precision. There were
              several hard misses. 

### yolo26s_1280_diverse_2  (+ ainew/pistol-csvic, all-to-train)  

- Config      : Added pistol-csvic (train 11,544). Same-source robbery/CCTV +
                posed pistols. 19 exact youtube-duplicate frames were removed.
- Hypothesis  : More CCTV data would increase recall.
- Result      : test/youtube R 0.351 (DOWN from 0.494), mAP50 0.284 (from 0.551),
              mAP50-95 0.071 (from 0.219).
- Conclusion  : dumping pistol-csvic data hurt recall. dataset contained several 
                frames from same video sources meaning similar images, hurting
                generalization.

### yolo26s_1280_diverse_downsample 
- Config      : Downsampled staged CCTV (GMD) randomly by 1/3. 
                Final train data:   6158
- Hypothesis  : Removing frames from the same sources should increase recall and 
                help with generalization.
- Result      : test/youtube R 0.512, mAP50 0.565, mAP50-95 0.283 (+0.064 vs the
              diverse run, +29% relative), R@0.05 0.594.
- Conclusion  : 27% less data improved recall from 0.351 to 0.512. Frames from the
              same scene were hurting generalization of the model.
### yolo26s_1280_diverse_downsample_2
- config      : In addition to downsampling pistol-csvic by 1/3. downsampled
                Open Images (large guns) by 50%. train data: 5195.
- Hypothesis  : Open Images (large guns) was biasing the model towards large-guns
                over smaller handguns (usual in CCTV/real-life)
- Result:     : test/youtube R 0.503, mAP50 0.528, mAP50-95 0.259 
                test openimages (mAP50 0.760 -> 0.689).
- Conclusion  : Large-gun bias was falsified. Downsampling OI (Open Images)
                by 50% hurt the diversity of the guns that could be detected by
                the model.

### yolo26s_1280_diverse_2_mixup
- config      : downsample staged CCTV (GMD) by 1/3 + mixup 0.1 -> 0.2. 
                train data: 6158
- Hypothesis  : increasing data diversity with mixup hyperparameter would increase 
                recall
- Result:     : test/youtube R 0.455 (Downsample experiment -> 0.512), mAP50 0.520,
                mAP50-95 0.265; P@0.2 also fell.
- Conclusion  : The hypothesis was falsified. Ambiguous samples can hurt recall 
                by making the model more conservative.

### yolo26s_1280_diverse_2_scale
- config      : downsample staged CCTV (GMD) by 1/3 + scale 0.9 -> 0.5. 
                train data: 6158
- Hypothesis  : shrink or enlarging samples by 0.9 could make the model detect overly
                small images
- Result:     : test/youtube R 0.464, mAP50 0.509 (Downsample experiment: -> 0.512)
- Conclusion  : Scaling to 0.9 made overly small images ambiguous, hurting training.

### yolo26s_1280_v2
- config      : Added two real-robbery-CCTV datasets:
                - mohammadreza/cctv-gun-detector (940 frames,
                  video-level deduped vs youtube -> dropped video V5).
                - footage-guntest (1419 frames). Evaluated against UGR and Hard-Negative frames. 
                - Train: 8517          
- Hypothesis  : Robbery frames from different sources would increase real/world      
                recall.
- Result      : Test/youtube P 0.879, R 0.544 - mAP 0.617, mAP-95 0.330
                UGR          P 0.899, R 0.804 - mAP 0.848, mAP50-95 0.636


- Conclusion  : Both youtube + UGR benchmark improved. Diverse CCTV footage
                from different video sources improved recall. (False-positive stayed at 24%)

### yolo26s_1280_v3
- Config       : v2 experiment + 496 hard-negative frames 
                 - Train: 9013
                 - 5.5% of training frames were hard-negatives (no firearms)
- Hypothesis   : Including hard-negatives in the dataset would increase precision
                 but slightly lower recall by making the model more conservative.
- Result       : Test/Youtube P 0.799, R 0.529 - mAP 0.585, mAP-95 0.290
                 UGR          P 0.899, R 0.794 - mAP50 0.856, mAP50-95 0.636 
                 
- Conclusion   : While P on Test/Youtube data decreased by 8% when compared to V2. 
                 a Hard negative Evaluation was made. Decreasing detection on non-firearm frames by 24%


### [HARD-NEGATIVE EVALUATION]
  confidence >=      Downsample     V2 (+diverse footage)     V3 (+hard neg)
  ----------------------------------------------------------------------
     0.05              37.0%          36.5%                2.2%
     0.10              28.2%          31.0%                1.8%
     0.20              22.5%          25.8%                1.2%
     0.25              22.2%          24.0%                1.0%
     0.30              21.0%          22.8%                1.0%
     0.50              17.8%          19.2%                0.5%

- Conclusion   : 24x Hard negatives by experiment v3.  

### Final Result

Three-model comparison (all @960px):
  Model                    youtube R   youtube mAP50-95   UGR mAP50   FP@0.25
  ----------------------------------------------------------------------------
  Downsample                 0.512          0.283           0.801      22.2%
  V2                         0.544          0.330           0.848      24.0%
  V3                         0.529          0.290           0.856       1.0%  

Speed: ~46 FPS (end-to-end) to ~85 FPS (inference) at 960px on an RTX 4060 laptop.

## USAGE
- SHIP MODEL V3: Best UGR mAP (0.856), Competitive out of distribution data/ low-quality frames(0.53). 24x lower false-positives. 
- use deploy_model.py to deploy to roboflow.
    $ python deploy_model.py
- use object_tracking.py to try model against .mp4 files
    $ object_predict.py src="path/to/.mp4"

## Reference & Acknowledgements

**Model / framework:** Ultralytics YOLO (YOLO26) — https://github.com/ultralytics/ultralytics

**Training datasets** (Roboflow Universe, CC BY 4.0 — attribution required):

| Role | Dataset | Author (workspace) | Source |
|------|---------|--------------------|--------|
| Staged CCTV | gun-detection-cctv | joe-workspace | https://universe.roboflow.com/joe-workspace/gun-detection-cctv/dataset/4 |
| Diverse positives | guns-r46kc | arads1 | https://universe.roboflow.com/arads1/guns-r46kc/dataset/2 |
| Real-CCTV positives | cctv-gun-detector | mohammadreza-anvari-h6ase | https://universe.roboflow.com/mohammadreza-anvari-h6ase/cctv-gun-detector/dataset/1 |
| Real-CCTV positives | footage-guntest | gun-proj5-workspace | https://universe.roboflow.com/gun-proj5-workspace/footage-guntest/dataset/2 |
| Hard negatives | security-footage-analysis | shauryaworkspace-guujq | https://universe.roboflow.com/shauryaworkspace-guujq/security-footage-analysis/dataset/1 |

Staged-CCTV footage derives from the **Gun Movies Database (GMD)**.

**Catalog images:** Open Images V7 (Google) — https://storage.googleapis.com/openimages/web/index.html

**Evaluation benchmark (not trained on):** UGR Handgun Detection dataset —
R. Olmos, S. Tabik, F. Herrera, *"Automatic Handgun Detection Alarm in Videos
Using Deep Learning,"* Neurocomputing 275 (2018) 66–72.
University of Granada, SCI2S — https://sci2s.ugr.es/weapons-detection

**Used during experiments but excluded from the final model:**
- pistol-csvic (ainew) — https://universe.roboflow.com/ainew/pistol-csvic/dataset/1 (regressed OOD recall; removed)
- phones-ayxme (yzrah) — https://universe.roboflow.com/yzrah/phones-ayxme/dataset/1 (downloaded, unused)

> The original datasets carry their own licenses (Roboflow sets here are CC BY 4.0).
> Verify terms before redistribution or commercial use.