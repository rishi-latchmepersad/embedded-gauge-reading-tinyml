# Training Diagram — evidence and history

Reviewed on 2026-10-13 for the **Training Diagram** page in
`process_diagrams.drawio`. This document records the evidence behind the page
and distinguishes image records, prepared examples and augmented samples.

## 1. Annotation corpus and provenance

The CVAT Images 1.1 exports in the WSL ML workspace were counted directly from
their `annotations.xml` records. Counts include records that a particular
trainer may reject because required shapes or image bytes are missing.

| Corpus | Training | Validation | Test | Total exported records |
|---|---:|---:|---:|---:|
| General gauges (`train_1`, `val_1`, `test_1`) | 7,326 | 916 | 915 | 9,157 |
| Small-gauge family (`train_2`, `val_2`, `test_2`) | 11 | 11 | 11 | 33 |

The temperature-gauge reference archives contain another **620 records**:

- `initial_temp_gauge/gauge_1_batch_1..8.zip`: **352 phone/PXL photographs**.
- `board_captures_1..4.zip`: **201 + 22 + 10 + 35 = 268 board-capture records**.
- The 22 records in `board_captures_2.zip` duplicate the refreshed `test_3.zip`;
  current preparation scripts explicitly exclude that archive from training.
  This leaves **598 eligible temperature-domain records**.

These are exported records, not a claim of that many independent physical
scenes. Some files are derived views or repeated captures. Do not add
`test_3.zip` to the above totals as another independent collection.

### What the exports establish

- General-gauge labels are `GaugeFace`, `Center`, `Tip`, `Start`, and `End`.
  Face labels are ellipses; imported landmarks are commonly boxes whose
  midpoints become coordinates.
- All **7,317 GaugeFace shapes** in `train_1.zip` have `source="auto"`.
  Its other **29,606 shapes** have `source="file"`, meaning imported annotations.
  The imported source flag does not identify their original annotator.
- `val_1.zip` has 865 automatic, 47 semi-automatic and 4 manual face shapes.
- Temperature archives use `temp_dial` ellipses, `temp_center` / `temp_tip`
  points and a `temp_c` attribute. These archives carry `source="manual"`.
  A CVAT source flag alone cannot establish how many labels were originally
  drawn by a person, reviewed, or created by an external assistant.
- Earlier notes explicitly describe manual visual review, reviewed bbox batches,
  manual center/tip manifests and AI-assisted board-label fill-in. Later
  archive counts therefore do not measure the original seed annotation effort.

### Automatic annotation implementation found

`D:/Projects/cvat/serverless/custom/gaugeface-yolo11m/nuclio/main.py` loads
an Ultralytics **YOLO11m** gauge detector and refines the detected region using
edge-contour ellipse fitting / Hough-circle fitting, with a box-based fallback.
The Nuclio handler returns **GaugeFace ellipse annotations only**. Its
`function.yaml` declares that single label. The combined/box/v2 variants
are present alongside it. `ml/scripts/run_cvat_teacher_keypoint_pipeline.py`
also identifies this detector as the CVAT teacher.

### Follow-up: the 450-image face seed and its split

A PC search located the local CVAT project at **`D:/Projects/cvat`** and the
original exports in `C:/Users/rishi_latchmepersad/Downloads`.

- `D:/Projects/cvat/first_450_training_filtered.zip` contains exactly **450
  images**, CVAT job 6 frames **0–449**, with **450 GaugeFace ellipses marked
  manual**. It matches the first 450 image records of the original XML exactly.
  This is strong evidence for the roughly 500-image manual face seed recalled
  by the user; the preserved filtered set is 450, not 500.
- The unfiltered `first_450_training.zip` and extracted
  `first_450_training/annotations.xml` actually contain **7,326 images**.
  Their name does not mean that they are already filtered. All 7,326 face shapes
  carry a manual flag, so that flag alone cannot measure the user's manual
  effort across the full export.
- The prepared `training/gaugeface_segmentation/dataset` and
  `detection_dataset` each contain **360 train / 90 validation images**.
  Every filename matches the filtered 450-image seed, with no missing records.
  The preparation scripts use an 80/20 split and default random seed 42.
  Ultralytics `.npy` image caches are excluded from these counts.
- The 450 records contain mostly imported Center/Tip/Start/End boxes. Manual
  edits are present: 5 center, 1 end, 2 start and 1 tip box. A manual face seed
  therefore does not imply that all its landmark annotations were drawn by hand.
- `true_value` and `gauge_id` attributes are present on the seed face shapes.
  There are 58 nonzero readings and 392 zeros. Zero can be a real reading or an
  unfilled/default value; the export alone cannot distinguish them. All 6,876
  records beyond the filtered seed repeat `true_value=0, gauge_id=28`, so those
  values must not be treated as verified readings for the full dataset.
- Saved detector `args.yaml` selects pretrained `yolo11m.pt`, 768-pixel input,
  100 requested epochs and patience 25. The 450-image seed trained the face
  detector; this is distinct from the final custom ellipse/keypoint models.
- Later `combined_gaugeface_training.zip` contains 1,365 images: 450 seed plus
  915 auto/semi-auto faces from the old test export. Its v2 contains 2,281:
  450 + 915 + 916. Their detector splits contain 1,092/273 and 1,824/457 images.
  These teacher-training splits reuse earlier validation/test collections and
  should not be confused with an independent final-model holdout.
- `Downloads/gaugeface_train_val_test` contains 11 train, 11 validation and
  11 test images. It is another small collection, separate from the 450 seed.

The exact seed image names and saved detector split membership are recorded in
[`training_manual_seed_450.csv`](training_manual_seed_450.csv).

**Still unconfirmed:** a VLM annotating face, center, tip and true reading.
The inspected notes, scripts and exports do not establish its identity/output.
The verified automatic face annotator is a CNN detector. No VLM model name is
asserted in the diagram. CVAT's Docker engine was not running during this
follow-up; these findings come from saved files, not a live database audit.

Inventory used for this review: `tmp/training_history_cvat_inventory.json`.
Follow-up inventories: `tmp/cvat_seed_inventory.json` and
`tmp/cvat_downloads_inventory.json`.

### Unique gauge identity audit

A follow-up XML audit found that the active training/validation/test archives
do not retain `gauge_id` attributes. The historical 450-image seed contains
27 distinct ID values (2–28), but these are not a reliable physical-gauge count:
ID 28 spans 51 original-name groups in the seed and repeats across the remaining
6,876 records in the full export. Some source-name groups also have multiple
IDs (`000090_png_jpg`: 9 and 10; `00027_jpg`: 11 and 12).

Removing the `.rf.` augmentation hash from filenames gives **1,216 source-name
groups** across the 9,157 general-gauge records, and **81 groups** within the
450-image seed. These are filename groups, not verified unique photographs or
physical gauges. Repeated photographs of one instrument, renamed copies and
derived views require an identity map to establish the physical-gauge total.
The current files therefore do not support an exact count of distinct gauges.
The audit is saved in `tmp/gauge_identity_inventory.json`.

### Visual estimate requested by the user

On 2026-10-13, all **1,216 general source-name representatives** were visually
reviewed on 19 indexed contact sheets, along with all 33 small-gauge images
and 35 sampled temperature-domain images. Representatives preferentially use
the old validation/test exports to reduce augmented training views. Face crops
retain a little housing; the index records each original filename and archive.

The subjective estimate is **approximately 750 visually distinct gauge-face
designs/variants**, with a broad plausible range of **600–900**. This is a
visual judgment, not an exhaustive identity assignment or a statistical
confidence interval. Obvious rotated, mirrored, relit and repeated views were
considered repeats; visibly different dial scales, branding and layouts were
considered distinct variants. Similar catalog gauges and low-resolution faces
create substantial ambiguity across sheets.

Examples of clear repeats include Viessmann faces among representatives
4/9/10/11, the two transmission-temperature images at 22/23, the long
`meter_a_*` sequence at 787–810, and `meter_b_*` at 811–818. The added
small-gauge collection contains three original-photo groups with many transformed
copies; the temperature samples mainly repeat a LittleGood thermometer design.
Separate physical instruments of the same design cannot reliably be identified
from these pictures. Do not describe the estimate as a verified physical-gauge
count or replace the exact image-record counts with it.

Contact sheets, full crops and `index.json` are in the WSL ML workspace at
`tmp/gauge_visual_review/`. The reproducible review generator is
`tmp/build_gauge_review_sheets.py` in the Windows workspace; it writes its
image outputs to the WSL workspace. The review did not change any source images
or annotations.

## 2. The final dataset sizes are stage-specific

### Ellipse localizer

Sources:

- `ml/scripts/train_ellipse_robust_384.py`: CVAT/image loader.
- `ml/scripts/train_ellipse_multiscale_universal_384.py`: final trainer.
- `tmp/run_iter8_wide_deep.sh`: actual launch recipe.
- `ml/artifacts/ellipse_iter8_universal_wide_deep/report.json`: saved report.

The final trainer pools the old generic train **and validation** archives,
then adds the tiny-gauge train/validation family and the 598 eligible
temperature-domain records. The independent test archives are evaluated
separately; the old generic validation set is not an independent holdout for
this final ellipse run.

| Component | Base eligible records | Repeat factor | Samples before augmentation |
|---|---:|---:|---:|
| Generic faces: 7,317 train + 916 validation | 8,233 | 1 | 8,233 |
| Tiny gauges: 11 train + 11 validation | 22 | 200 | 4,400 |
| Temperature domain, excluding test-3 duplicate | 598 | 4 | 2,392 |
| Total | **8,853** | — | **15,025** |

The augmentation routine adds one transformed view per sample, producing
**30,050 training samples**, matching the saved report. It uses scale,
translation, contrast and random horizontal flip with aligned ellipse targets.
The 30,050 figure does not represent 30,050 independently labelled photographs.

### Keypoint model

Sources:

- `ml/scripts/prepare_gauge_keypoint_224_data.py`.
- `ml/data/gauge_keypoint_224/{train,val,test}/center.npy` and `tip.npy`.
- `ml/scripts/train_keypoint_unet_224.py`.
- `tmp/run_keypoint_wide_aug.sh`.

Prepared-array headers establish **7,922 train / 925 validation / 946 test**
examples, with each channel shaped `N x 56 x 56`. Preparation requires all
three annotations—face ellipse, center and tip—and available image data.

Unlike the final ellipse run, this preparation retains `val_1/val_2` as
validation. Tests come from `test_1/test_2/test_3`; unusable records make
stage-specific counts differ from the raw CVAT counts.

The loader converts to grayscale, crops around the labelled face ellipse at
**1.35x** expansion, resizes to **224x224**, remaps coordinates and renders
Gaussian **56x56** center/tip targets. The wide-augmentation launcher doubles
7,922 examples to **15,844** using flip, rotation, scale and brightness changes.
Transformed coordinates are used to regenerate heatmaps; heatmaps are not
simply rotated as images.

## 3. Architecture history and the reasons for changing direction

This is a sequence of research families, not a claim that every family used
the same dataset, split, loss, export path or evaluation protocol.

| Period / family | Work recorded | Lesson carried forward |
|---|---|---|
| Classical CV comparator | Hough circle/line methods, radial/polar edge votes, rim fitting and template/ray alternatives | Geometry provides an interpretable baseline; glare and framing require conservative gates. This branch needs no neural-network training. |
| Early scalar / rectified reader | ImageNet-pretrained MobileNetV2 regressors; OBB/rectifier plus scalar chains; hard-case fine-tunes | Crop, calibration and domain mismatch could dominate the final reading. A scalar score did not expose the failing geometry. |
| Explicit geometric heads | MobileNetV2 keypoints/direction; coordinate regression, CoordConv, polar angle CNNs and classifiers | Some pooled heads collapsed toward a fixed pose or lost fine spatial information. Gauge-specific angle/value calibration matters. |
| Face localizer research | YOLO/YOLO11 OBB, compact OBB, joint MobileNetV2 OBB+center, QARepVGG Mini/Pro and teacher/student distillation | A good crop is necessary; precision, activation memory, supported operators and exportability all influence selection. |
| Center/tip research | DS-CNN heatmaps; SimCC and spatial SimCC; KD SimCC; HRNet-lite/ECA; heatmaps plus DARK; RepVGG and CoordConv candidates | Compare center, tip, angle and physical reading errors. Tip localization was often the limiting stage; retaining spatial features helped. |
| U-Net iterations | MobileNetV2 U-Net experiments, larger and reduced-input variants, then custom QAT-friendly U-Nets | Skip connections preserve detail; matched crops and regenerated augmentation targets matter. Early clean-split wins did not establish board generalization. |
| Current custom pair | Universal multiscale ellipse localizer + wide grayscale center/tip U-Net | Explicit geometry, int8/QAT export and on-board preprocessing/memory parity form the current deployment contract. |

Early transfer-learning experiments and later custom builders should not be
conflated. The current builders create new Keras models without loading
ImageNet backbones. QAT itself is not transfer learning. Vision transformers
appear as a project goal / suggested future direction; the reviewed evidence
does not establish a transformer as the deployed model or a completed winner.

## 4. Current training recipes and firmware bindings

### `gauge_ellipse_v1.0`

- Artifact: `ellipse_iter8_universal_wide_deep/model_int8.tflite`.
- Trainer: `train_ellipse_multiscale_universal_384.py`.
- Custom six-stage Conv/BN/ReLU encoder with channels **24,32,48,64,96,128**.
- **384x384 grayscale** input; shared multiscale **24/48/96** center/rim maps,
  geometry heads and learned scale-confidence selection.
- Supervision: center/rim maps, center/radii geometry and scale selection.
- Launcher: **25 FP32 epochs + 10 QAT epochs**, with repeats/augmentation above.
- Firmware wrapper: `ai_network_mobilenetv2_center_detector.c` actually binds
  `ellipse_iter8_universal_wide_deep_int8`; its historical filename is not an
  indication that the active architecture is MobileNetV2.

### `gauge_keypoint_v1.0`

- Artifact: `keypoint_unet_224g_wide_aug/model_int8.tflite`.
- Builder: `ml/src/embedded_gauge_reading_tinyml/keypoint_unet_224.py`.
- Trainer: `train_keypoint_unet_224.py`; launcher `run_keypoint_wide_aug.sh`.
- Custom Conv/BN/ReLU encoder–decoder with skip concatenations, width **alpha 1.5**,
  **2,238,386 parameters**, **224x224x1** input and **56x56x2** sigmoid heatmaps.
- Focal heatmap loss weights the harder tip channel **1.5x** versus center.
- Launcher: **60 FP32 epochs + 20 QAT epochs**, batch 16, augmentation enabled.
- Firmware wrapper: `ai_network_mobilenetv2_rectifier_hardcase_finetune.c`
  actually binds `keypoint_unet_224g_wide_aug_int8`.

The current learned path computes geometry first. Center-to-tip angle is
converted to a physical value by the selected gauge calibration profile;
this final conversion is deterministic rather than a new scalar network.
Earlier scalar and some geometry variants used true reading / temperature
labels as supervision. Current ellipse/keypoint targets are geometric.

## 5. Evaluation, export and the deployment feedback loop

- Keep test domains separate and report center/tip/radius errors, angle error,
  physical reading error, rejection/failure rates and worst-case examples.
- Run the complete localizer → crop → keypoint → profile conversion, because
  a good keypoint result on a labelled crop does not prove predicted-crop accuracy.
- Compare FP32/QAT/int8 predictions and tensor contracts before board packaging.
- Export QAT int8 TFLite; compile/package with ST Edge AI / Cube.AI for the N6.
  Check supported operators, activation/pool layout and external-flash signatures.
- Validate actual board captures and saved-image replay against firmware resize,
  crop, quantization and decoder behavior. Feed bad captures and reviewed labels
  back into the next training iteration.

The keypoint `report.json` values **3.66 px center / 6.28 px tip** come from the
trainer's **first 100 validation examples**, measured in 224-pixel crop space.
They are not a universal end-to-end test guarantee. `per_split_report.json`
contains different test-domain results (including substantially worse tails).
This review did not rerun evaluation or retrain any model.

## 6. Notes reviewed

Windows and WSL copies were considered, with final model builders/reports and
Windows firmware bindings used to resolve stale “current” statements:

- `docs/ai-memory.md` — full chronology search and relevant training/data sections.
- `docs/ai-memory/archive.md` — earlier scalar, geometry and deployment history.
- `docs/ai-memory/ml-experiments.md` — classical, scalar and geometry experiments.
- `docs/ai-memory/foundation.md`, `workflow.md`, `firmware-board.md`.
- `docs/ai-memory/polar-vote-topk-2026-05-17.md`.
- `docs/ai-memory/polar-vote-deploy-2026-05-18.md`.
- `docs/ai-memory/2026-08-02-firmware-baseline-redesign.md`.
- `docs/CANONICAL_DATA_BUILD_SUMMARY.md`, `anchored_summary.md`,
  `RANGE_AWARE_TRAINING.md` and the baseline summaries.

The older 141-row canonical build and 433/468/557-row geometry manifests are
specific historical subsets. They should not be presented as the entire final
corpus. Likewise the older OBB + tip-focus-v18 description in AGENTS/foundation
notes is superseded for this diagram by the inspected ellipse/keypoint bindings.
