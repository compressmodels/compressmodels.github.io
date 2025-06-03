---
layout: post
title: "MLP"
date: 2025-06-01
author: "Sam Randall"
---

## Multilayer Perceptron Speedup

Accelerating Wide MLPs with Early Exit Inference: A Case Study on Breast Cancer Classification

We present an empirical evaluation of a custom ONNX-based inference acceleration framework applied to wide multilayer perceptrons (MLPs). Our method integrates early exit mechanisms within ONNX-exported MLP classifiers to significantly reduce inference time without compromising accuracy.

### Dataset
We evaluate our method on the [Breast Cancer Wisconsin (Diagnostic) dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), a well-known benchmark for binary classification of tabular medical data. The dataset contains 569 data points and 30 features.

### Experimental Setup

Model: One-Layer `MLPClassifier` (scikit-learn)

Export format: ONNX

Baseline: Standard ONNX runtime inference

Accelerated: ONNX + Early Exit (custom acceleration logic)

Metrics: Inference latency (s), training/test accuracy

Hidden layer size (H): Varied from 16 to 8192

### Methods

We take a model and consider its training dataset, define geometry-based rules.
At inference time given a *single* data point, the new algorithm first checks
whether the datapoint satisfies any of the geometry-based rules (is within a hypersphere, is within a halfspace, etc.).
If it does, we output a prediction specific to the relevant rule. If no rules contain the datapoint being evaluated, we run the model as usual.

### Results

|Hidden Layer Size   |ONNX (s)|ONNX + Early Exit (s) |Improvement|ONNX Accuracy (Test) |ONNX + Early Exit Accuracy (Test)|
|--------------------|--------|----------------------|-----------|---------------------|---------------------------------|
|1024                |0.006377|0.004472              |1.4258x    |0.9386               |0.9386                           |
|2048                |0.007553|0.004979              |1.5168x    |0.9298               |0.9123                           |
|4096                |0.00918 |0.005332              |1.7215x    |0.9123               |0.9123                           |
|8192                |0.012255|0.007714              |1.5887x    |0.9649               |0.9649                           | 

For a given hidden layer size, we compute the average time of running inference on a single sample. For 1000 samples, with ONNX, an MLP with a hidden layer size of 4096 would take 9.18 seconds to run. With ONNX and the custom acceleration, it would take 5.33 seconds to run those 1000 samples.

### Key Observations
- Significant latency reduction was observed at high hidden dimensions (H ≥ 1024), where computational complexity is highest.
- Algorithm did not universally result in speedups in low H regime.
- Minimal degradation in test accuracy across most model sizes.
- Our method performs best with very wide architectures, confirming its suitability for deployment scenarios involving large-capacity models.

### Stay Tuned For:
- An optimized algorithm so that it achieves latency improvement in smaller H settings. 
- A notebook demo so you can see the software in use to achieve this gains.
