# SEAS TOG Adversarial Attacks

Students think and complete the main algorithms for the TOG attacks. Fill in the TODO.

## Introduction to TOG

TOG, or **Targeted Objectness Gradient**, is a family of adversarial attacks designed specifically for object detectors.

Unlike image-classification attacks, an object-detection attack may need to influence several parts of the detector simultaneously, including:

- whether an object is detected,
- where an object is localized,
- which class is assigned to an object,
- and how many detections are produced.

This repository includes four TOG attack variants:

1. **Object vanishing**

   Reduces detector objectness so that real objects disappear from the detector output.

2. **Object fabrication**

   Increases detector objectness in background regions, encouraging false detections.

3. **Object mislabeling**

   Preserves detected objects but changes their predicted class labels toward selected target classes.

4. **Untargeted attack**

   Increases the detector loss for the original benign detections without specifying an exact target output.

Each attack uses an iterative signed-gradient update. After every update, the adversarial image is projected back into:

- the valid image range `[0, 1]`, and
- an L-infinity perturbation region around the original image.

The maximum allowed pixel perturbation is controlled by `epsilon`.

## Repository structure

- `yolov3_model.py`  
  YOLOv3 network definition, Darknet weight loading, output decoding, detection, and non-maximum suppression.

- `tog_model.py`  
  YOLOv3 target encoding, attack losses, and gradient calculations required by TOG.

- `tog_attacks_student.py`  
  Student template containing incomplete implementations of the four main TOG algorithms.

- `utils.py`  
  COCO class names, image preprocessing, visualization, and mislabeling-target generation.

- `main.ipynb`  
  Runnable examples for loading the detector and evaluating the attacks. 

- `main_all.ipynb`  
  All in one file

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt