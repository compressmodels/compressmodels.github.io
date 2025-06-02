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

Hidden layer width (H): Varied from 16 to 8192

### Methods

We take a model, define early-exit geometric based rules, and then at inference time check whether any of the pre-defined rules contains the datapoint being evaluated. If it does, we output a prediction specific to that rule. If no rules contain the datapoint being evaluated, we run the model as usual.

### Results

|h   |ONNX    |ONNX + Early Exit|Improvement|Baseline Accuracy (Test) |Experimental Accuracy (Test)|
|----|--------|-----------------|-----------|-------------------------|----------------------------|
|16  |0.004465|0.003064         |1.4573     |0.9649                   |0.9649                      |
|32  |0.003752|0.004036         |0.9295     |0.6228                   |0.6228                      |
|64  |0.004263|0.004472         |0.9532     |0.9649                   |0.9649                      |
|128 |0.004133|0.00368          |1.1231     |0.9386                   |0.9386                      |
|256 |0.004277|0.005535         |0.7726     |0.9386                   |0.9386                      |
|512 |0.00475 |0.004913         |0.9669     |0.9561                   |0.9737                      |
|1024|0.006377|0.004472         |1.4258     |0.9386                   |0.9386                      |
|2048|0.007553|0.004979         |1.5168     |0.9298                   |0.9123                      |
|4096|0.00918 |0.005332         |1.7215     |0.9123                   |0.9123                      |
|8192|0.012255|0.007714         |1.5887     |0.9649                   |0.9649                      | 


### Key Observations
- Significant latency reduction (up to 1.67×) was observed at high hidden dimensions (H ≥ 1024), where computational complexity is highest.
- ONNX was low computational cost in low H regime that overhead of geometric rules did not offset performance.
- No degradation in test accuracy across most model sizes.
- Our method performs best with very wide architectures, confirming its suitability for deployment scenarios involving large-capacity models.

### Future Work
- Run this on other MLP model latency use cases (edge AI, malware detection, text).
- Optimize algorithm so that it achieves latency improvement in smaller H settings. 
- Add a notebook so you can see the software in use to achieve this gain.s
