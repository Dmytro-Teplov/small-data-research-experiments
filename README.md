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