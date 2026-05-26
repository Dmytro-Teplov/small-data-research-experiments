# Experiments for Low-Data Object Detection with YOLOv12

This repository contains experiment scripts, dataset preparation utilities, result extraction tools, and analysis notebooks for the thesis:

**An Empirical Study of Training Strategies for Object Detection with Limited Data**  
Dmytro Teplov, Master Graduation Thesis, Vilnius Gediminas Technical University, 2026

The project evaluates how object detection models behave when only a small amount of labelled target-domain data is available. The experiments focus on YOLOv12-based object detection and compare training from scratch, data augmentation, COCO-based transfer learning, custom pretraining, freezing strategies, model capacity, dataset reduction, and synthetic-domain pretraining.

---

## Research Goal

The goal of this repository is to support a controlled empirical study of training strategies for object detection under extremely limited-data conditions.

The thesis investigates the following core question:

> How do dataset size, pretraining quality, model capacity, augmentation, and fine-tuning strategy affect object detection performance under extremely limited labelled data conditions?

The repository is intended to make the experimental pipeline easier to reproduce, extend, and analyze.

---

## Main Experimental Factors

The experiments are organized around the following variables:

| Factor | Values / Description |
|---|---|
| Model family | YOLOv12 |
| Model sizes | `n`, `s`, `m`, `l` and, for COCO-scale analysis, `x` |
| Target task | Trash object detection |
| Target classes | `metal_can`, `disposable_cup`, `styrofoam` |
| Target dataset fractions | `30%`, `50%`, `80%`, `100%` |
| Pretraining sources | None, official COCO weights, custom COCO-subset weights, synthetic dataset weights |
| COCO pretraining fractions | `10%`, `30%`, `50%`, `80%`, plus official full-COCO weights |
| Fine-tuning strategy | Full fine-tuning or frozen backbone/neck with detection-head training |
| Augmentation | YOLO built-in augmentation enabled or disabled |
| Main metric | `mAP50-95` |

---

## Repository Structure

A recommended layout for this repository is shown below. Adjust names if your local implementation differs.

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── data/
│   │   ├── trash_full.yaml
│   │   ├── trash_80.yaml
│   │   ├── trash_50.yaml
│   │   ├── trash_30.yaml
│   │   └── synthetic.yaml
│   ├── experiments/
│   │   ├── scratch_no_aug.yaml
│   │   ├── scratch_aug.yaml
│   │   ├── coco_official_freeze.yaml
│   │   ├── coco_official_full_finetune.yaml
│   │   ├── coco_custom_freeze.yaml
│   │   └── synthetic_pretrain.yaml
│   └── paths.yaml
├── data/
│   ├── README.md
│   ├── trash/
│   │   ├── full/
│   │   ├── split_80/
│   │   ├── split_50/
│   │   └── split_30/
│   ├── coco_subsets/
│   └── synthetic/
├── scripts/
│   ├── prepare_trash_dataset.py
│   ├── create_dataset_subsets.py
│   ├── generate_synthetic_dataset.py
│   ├── train.py
│   ├── train_batch.py
│   ├── evaluate.py
│   ├── extract_results.py
│   └── aggregate_results.py
├── notebooks/
│   ├── 01_progressive_coco_reduction.ipynb
│   ├── 02_baseline_experiments.ipynb
│   ├── 03_pretraining_capacity.ipynb
│   ├── 04_domain_mismatch.ipynb
│   └── 05_per_class_analysis.ipynb
├── results/
│   ├── raw/
│   ├── processed/
│   ├── tables/
│   └── figures/
├── weights/
│   ├── official/
│   ├── coco_custom/
│   └── synthetic/
└── docs/
    └── thesis_summary.md
```

Large files such as datasets, trained weights, and raw training outputs should usually not be committed directly. Use `.gitignore`, release assets, cloud storage, or a data registry where appropriate.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` may include:

```text
ultralytics
opencv-python
numpy
pandas
matplotlib
pyyaml
tqdm
scikit-learn
jupyter
```

Depending on your YOLOv12 implementation, you may need to install the YOLOv12 package directly from its source repository or include it as a local submodule.

---

## Datasets

### COCO

COCO is used as the large-scale source dataset for pretraining and progressive dataset reduction experiments. The thesis uses COCO because it is a standard object detection benchmark with diverse categories, complex scenes, and enough samples to support controlled subsampling.

Expected COCO subset structure:

```text
data/coco_subsets/
├── coco_10/
├── coco_30/
├── coco_50/
├── coco_80/
└── coco_100/
```

### Custom Trash Dataset

The target low-data task uses a manually curated trash object detection dataset based on selected TACO images and custom images.

Classes:

```text
0 metal_can
1 disposable_cup
2 styrofoam
```

The full dataset contains 201 images and is split into smaller fractions to simulate constrained target-domain data availability.

Expected structure:

```text
data/trash/full/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

YOLO dataset YAML example:

```yaml
path: data/trash/full
train: images/train
val: images/val
test: images/test

names:
  0: metal_can
  1: disposable_cup
  2: styrofoam
```

### Synthetic Dataset

A synthetic dataset is used to study domain-mismatched transfer learning. It is generated with deterministic Python scripts and is designed to teach broad objectness, color variation, shape variation, spatial relationships, and background rejection.

Example command:

```bash
python scripts/generate_synthetic_dataset.py \
  --output data/synthetic \
  --num-images 10000 \
  --num-classes 3 \
  --seed 42
```

---

## Experiment Groups

### 1. Training from Scratch

Purpose: establish whether YOLOv12 can learn the target task without pretrained weights.

```bash
python scripts/train.py \
  --model yolo12n.yaml \
  --data configs/data/trash_full.yaml \
  --pretrained false \
  --augment false \
  --project results/raw/scratch_no_aug
```

### 2. Training from Scratch with Augmentation

Purpose: test whether augmentation alone can compensate for limited data.

```bash
python scripts/train.py \
  --model yolo12n.yaml \
  --data configs/data/trash_full.yaml \
  --pretrained false \
  --augment true \
  --project results/raw/scratch_aug
```

### 3. Official COCO Transfer Learning

Purpose: evaluate the effect of official COCO-pretrained YOLOv12 weights.

```bash
python scripts/train.py \
  --model weights/official/yolo12n.pt \
  --data configs/data/trash_full.yaml \
  --freeze 10 \
  --augment true \
  --project results/raw/coco_official_freeze
```

For full fine-tuning:

```bash
python scripts/train.py \
  --model weights/official/yolo12n.pt \
  --data configs/data/trash_full.yaml \
  --freeze 0 \
  --augment true \
  --project results/raw/coco_official_full_finetune
```

### 4. Custom COCO-Subset Transfer Learning

Purpose: measure how the amount of pretraining data affects downstream performance.

```bash
python scripts/train_batch.py \
  --weights-dir weights/coco_custom \
  --data-dir configs/data \
  --models n s m l \
  --pretrain-fractions 10 30 50 80 \
  --target-fractions 30 50 80 100 \
  --freeze 10 \
  --repeats 5 \
  --project results/raw/coco_custom
```

### 5. Synthetic Pretraining / Domain Mismatch

Purpose: compare domain-mismatched synthetic pretraining against random initialization and COCO pretraining.

```bash
python scripts/train_batch.py \
  --weights-dir weights/synthetic \
  --data-dir configs/data \
  --models n s m l \
  --target-fractions 30 50 80 100 \
  --freeze 10 \
  --repeats 3 \
  --project results/raw/synthetic_pretrain
```

---

## Result Extraction and Analysis

After experiments finish, extract metrics from YOLO output folders:

```bash
python scripts/extract_results.py \
  --runs-dir results/raw \
  --output results/processed/all_results.csv
```

Aggregate repeated runs:

```bash
python scripts/aggregate_results.py \
  --input results/processed/all_results.csv \
  --output results/tables/summary.csv \
  --group-by experiment model pretraining_fraction target_fraction freeze augment
```

Typical columns in the processed results file:

```text
experiment,model,pretraining_source,pretraining_fraction,target_fraction,freeze,augment,run_id,best_map50_95,best_map50,precision,recall,class,per_class_map50_95
```

---

## Key Findings Reflected by the Scripts

The experiment and analysis scripts are designed to reproduce the main findings from the thesis:

1. Training from scratch performs poorly under extreme target-data scarcity.
2. Augmentation improves robustness but does not replace pretrained visual representations.
3. COCO-pretrained YOLOv12 models substantially outperform randomly initialized models.
4. Freezing pretrained layers improves stability, although full fine-tuning can sometimes achieve higher peak performance.
5. Compact and medium-sized models may provide a better balance between adaptability and stability than larger models in very small target datasets.
6. Larger and more relevant pretraining datasets generally improve transfer learning, but gains are not always linear.
7. Synthetic pretraining improves over random initialization but remains weaker than domain-relevant COCO pretraining.
8. Evaluation on very small validation or test splits can produce inflated metrics because reduced subsets may be less diverse and less difficult.

---

## Metrics

The main evaluation metric is **mAP50-95**, which averages mean Average Precision over IoU thresholds from 0.50 to 0.95 in steps of 0.05.

Additional metrics may include:

- `mAP50`
- precision
- recall
- per-class `mAP50-95`
- standard deviation across repeated runs
- qualitative error analysis of false positives and false negatives

---

## Reproducibility Notes

To make results easier to compare:

- Keep train/validation/test splits fixed for each target dataset fraction.
- Store split-generation seeds.
- Run each experiment multiple times.
- Save full YOLO training logs.
- Save `results.csv`, `args.yaml`, model weights, and plots for each run.
- Report mean and standard deviation, not only the best run.
- Interpret metrics cautiously when validation or test sets are very small.

GPU non-determinism, mixed precision, small batch sizes, and batch normalization statistics may cause small run-to-run differences even when seeds are fixed.

---

## Hardware

Initial pipeline validation was performed on local hardware:

```text
CPU: Intel Core i5-8300H @ 2.3 GHz
RAM: 16 GB
GPU: NVIDIA GeForce GTX 1050, 4 GB VRAM
```

Larger experiments were designed for Google Colab or equivalent GPU environments, including A100-class hardware.

---

## Example Workflow

```bash
# 1. Prepare the trash dataset splits
python scripts/prepare_trash_dataset.py --input data/raw_taco --output data/trash/full
python scripts/create_dataset_subsets.py --input data/trash/full --output data/trash --fractions 30 50 80 100 --seed 42

# 2. Generate synthetic pretraining data
python scripts/generate_synthetic_dataset.py --output data/synthetic --num-images 10000 --seed 42

# 3. Run baseline experiments
python scripts/train_batch.py --config configs/experiments/scratch_no_aug.yaml
python scripts/train_batch.py --config configs/experiments/scratch_aug.yaml

# 4. Run transfer learning experiments
python scripts/train_batch.py --config configs/experiments/coco_official_freeze.yaml
python scripts/train_batch.py --config configs/experiments/coco_custom_freeze.yaml
python scripts/train_batch.py --config configs/experiments/synthetic_pretrain.yaml

# 5. Extract and aggregate results
python scripts/extract_results.py --runs-dir results/raw --output results/processed/all_results.csv
python scripts/aggregate_results.py --input results/processed/all_results.csv --output results/tables/summary.csv

# 6. Open notebooks for visualization and discussion
jupyter notebook notebooks/
```

---

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
.venv/
.env

# Jupyter
.ipynb_checkpoints/

# Datasets and large artifacts
data/coco_subsets/
data/trash/
data/synthetic/
weights/
runs/
results/raw/
*.pt
*.onnx
*.engine

# System files
.DS_Store
Thumbs.db
```

---

## Citation

If you use this repository, cite the thesis:

```bibtex
@mastersthesis{teplov2026lowdataobjectdetection,
  author  = {Teplov, Dmytro},
  title   = {An Empirical Study of Training Strategies for Object Detection with Limited Data},
  school  = {Vilnius Gediminas Technical University},
  year    = {2026},
  type    = {Master Graduation Thesis}
}
```

---

## License

Add a license before public release. For research code, common options include MIT, Apache-2.0, or BSD-3-Clause. Dataset files may be subject to separate licenses from COCO, TACO, or other sources.

---

## Acknowledgements

This repository supports the experimental work conducted for the master thesis under the supervision of Artur Radzivil. Some COCO progressive dataset reduction results were analyzed from raw experimental outputs provided as part of related research conducted with Artur Radzivil and PhD Andrej Bugajev.
