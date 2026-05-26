# Experiments for Low-Data Object Detection with YOLOv12

This repository contains experiment scripts, dataset preparation utilities, result extraction tools, and analysis notebooks developed for the thesis:

**An Empirical Study of Training Strategies for Object Detection with Limited Data**  
Dmytro Teplov, Master Graduation Thesis  
Vilnius Gediminas Technical University, 2026

The project evaluates how YOLOv12-based object detection models behave when only a small amount of labelled target-domain data is available. The experiments compare training from scratch, data augmentation, COCO-based transfer learning, custom pretraining, freezing strategies, model capacity, target dataset reduction, and synthetic-domain pretraining.

---

Full result folders, including trained model outputs and experiment logs, are stored separately in Google Drive.

> **Google Drive results archive:**  
> Google Drive link: `https://drive.google.com/drive/folders/1nzIkq9CWhbB9Hmah8EMoh0BGGSIZkzWb?usp=sharing`

The Google Drive archive follows the same directory structure shown below, so the analysis scripts can be used with either the local repository files or the downloaded result folders.

---

## Research Goal

The goal of this repository is to support a controlled empirical study of training strategies for object detection under extremely limited-data conditions.

The thesis investigates the following research question:

> How do dataset size, pretraining quality, model capacity, augmentation, and fine-tuning strategy affect object detection performance under extremely limited labelled data conditions?

The repository is intended to make the experimental pipeline easier to reproduce, inspect, extend, and analyze.

---

## Main Experimental Factors

The experiments are organized around the following variables:

| Factor | Values / Description |
|---|---|
| Model family | YOLOv12 |
| Model sizes | `n`, `s`, `m`, `l`; for COCO-scale analysis also `x` |
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

```text
Experiments/
├── 1.variative_zero_experiments/
│   └── Experiments with randomly initialized YOLOv12 models
│
├── 2.variative_official_experiments/
│   └── Experiments using official COCO-pretrained YOLOv12 weights
│
├── 3.variative_cust_experiments/
│   └── Experiments using custom COCO-subset-pretrained weights
│
├── 4.variative_synt_experiments/
│   └── Experiments using synthetic-domain pretrained models
│
├── 5.variative_per_class/
│   └── Per-class metric extraction and analysis
│
├── Datasets/
│   └── Dataset configuration files and dataset preparation utilities
│
└── Pretraining_Models/
    └── Custom pretrained YOLOv12 model weights