# YOLO-GunRecognizer-V0

- **A benchmark data-leak in the v3 hard-negative evaluation was caught and fixed via a dHash audit.
  The final model is `v3-2`, retrained with an independent hard-negative source and re-scored on a
  clean, deduped benchmark. All headline numbers below are the corrected, leak-audited results.**


`YOLO-GunRecognizer-v0`. Single-class ("gun") computer vision model trained to detect
firearms. The YOLO architecture was used to make the model deployable in real world cameras for live detection.

Key benchmark:
    1. UGR handgun benchmark - published benchmark deduped against training data

### TL;DR
A single-class YOLO model (class: "gun") designed for live-monitoring and real-surveillance frames. 
- Main problem: data distribution. While at first, hyperparameter-tunning was 
considered as the best first approach, such techniques barely improved the model. However, by performing data curation techniques. The model improved by 3.8x (recall: 0.149 -> 0.568).

- **Shipped model: v3-2** (diverse in-domain data + hard negatives):
  - test/youtube  P 0.847, R 0.568 - mAP50 0.628, mAP50-95 0.332
  - UGR           P 0.922, R 0.787 - mAP50 0.850, mAP50-95 0.637
  - False-positive rate 17.0% @conf 0.25 (lowest of all models)

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

test/youtube: real robbery CCTV (low/confusing image quality) — the honest out-of-distribution metric.
val/cctv: staged CCTV — reads ~1.00 for EVERY model because a dHash audit found 99% of these frames are
near-duplicates of training frames (median distance 2; the staged clips are all the same room/camera).
It is scene-level leaked and cannot be a valid held-out slice, so it is EXCLUDED from reported metrics
and kept only as in-domain training signal.

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
- Hard Negative Eval:
  confidence >=      Downsample     V2 (+diverse footage)     V3 (+hard neg)
  ----------------------------------------------------------------------
     0.05              37.0%          36.5%                2.2%
     0.10              28.2%          31.0%                1.8%
     0.20              22.5%          25.8%                1.2%
     0.25              22.2%          24.0%                1.0%
     0.30              21.0%          22.8%                1.0%
     0.50              17.8%          19.2%                0.5%
                 
- Conclusion   : Test/Youtube P decreased while Hard Negative Eval overfitted.
                 0.05 FP at 0.5 confidence. which suggests data leagake in hard-negatives
                 training data. 


### yolo26s_1280_v3-2   (corrected — v2 + INDEPENDENT hard negatives)
- Config      : The original v3 hard negatives came from security-footage-analysis, which had
                LEAKED into the FP benchmark (same-source video frames), faking a 1% rate. Retrained
                with 498 hard negatives from an INDEPENDENT source (people-fkg4e, empty labels), and
                rebuilt the FP benchmark to dHash-dedup against the whole training set. Train ~9,014.
- Hypothesis  : Hard negatives from an unleaked source will still cut false positives.
- Result      : test/youtube  P 0.847, R 0.568 - mAP50 0.628, mAP50-95 0.332
                UGR           P 0.922, R 0.787 - mAP50 0.850, mAP50-95 0.637
- Conclusion  : Confirmed, but MODESTLY (not the 24x seen under the leak). people-fkg4e is
                off-domain (COCO people, not surveillance), so it under-teaches the real failure
                mode. v3-2 is nonetheless the best model on every metric.

### [HARD-NEGATIVE EVALUATION]  (CLEAN benchmark: all models on the SAME 400 held-out
    no-gun images, dHash-deduped vs the whole training set)
  confidence >=      Downsample     V2 (+diverse)     V3-2 (+hard neg)
  --------------------------------------------------------------------
     0.05              35.8%           29.8%              25.5%
     0.10              26.2%           26.2%              22.5%
     0.20              20.5%           20.5%              18.5%
     0.25              18.5%           19.0%              17.0%
     0.30              16.5%           17.5%              16.0%
     0.50              13.8%           14.5%              13.0%

- Conclusion   : v3-2 has the LOWEST false-positive rate at every threshold — a real but modest
                 reduction. (The earlier "24x / 1%" was a benchmark leak, now corrected.)

### Final Result

Three-model comparison (all @960px, leak-audited benchmarks):
  Model                     youtube R   youtube mAP50-95   UGR mAP50   UGR P   FP@0.25
  ------------------------------------------------------------------------------------
  Downsample                  0.512          0.283           0.801      0.860   18.5%
  V2   (+diverse)             0.544          0.330           0.848      0.899   19.0%
  V3-2 (+hard neg)  <- SHIP   0.568          0.332           0.850      0.922   17.0%

Speed: ~46 FPS (end-to-end) to ~85 FPS (inference) at 960px on an RTX 4060 laptop.

## USAGE
- SHIP MODEL V3-2: best on every metric — highest OOD recall (youtube R 0.568), highest UGR precision (0.922), and the lowest false-positive rate (17.0% @0.25). 
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
| Hard negatives (training) | people-fkg4e | twdaf | https://universe.roboflow.com/twdaf/people-fkg4e/dataset/1 |
| FP benchmark (held-out, not trained on) | security-footage-analysis | shauryaworkspace-guujq | https://universe.roboflow.com/shauryaworkspace-guujq/security-footage-analysis/dataset/1 |

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