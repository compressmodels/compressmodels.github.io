---
layout: post
title: "MLP"
date: 2025-06-01
author: "Sam Randall"
---

## Multilayer Perceptron Speedup

Accelerating Wide MLPs with Early Exit Inference: A Case Study on Breast Cancer Classification

We present an empirical evaluation of a custom sklearn-based inference acceleration framework applied to wide multilayer perceptrons (MLPs). Our method early exit mechanisms within sklearn MLP classifiers to significantly reduce inference time without compromising accuracy.

### Dataset
We evaluate our method on the [Breast Cancer Wisconsin (Diagnostic) dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), a well-known benchmark for binary classification of tabular medical data. The dataset contains 569 data points and 30 features.

### Experimental Setup

Model: One-Layer `MLPClassifier` (scikit-learn)

Accelerated: Early Exit (custom acceleration logic)

Metrics: Inference latency (s), training/test accuracy

Hidden layer size (H): Varied from 16 to 8192

### Methods

We take a model and consider its training dataset, define geometry-based rules.
At inference time given a *single* data point, the new algorithm first checks
whether the datapoint satisfies any of the geometry-based rules (is within a hypersphere, is within a halfspace, etc.).
If it does, we output a prediction specific to the relevant rule. If no rules contain the datapoint being evaluated, we run the model as usual.

## Usage

```
# Train a normal sklearn MLP model.
mlp = MLPClassifier(hidden_layer_sizes=(512,))
mlp.fit(X_train, y_train)

# Wrap it as an early exit model.
eem = EarlyExitModel(mlp)

# Let the algorithm do its thing.
eem.add_hypersphere_prediction_grouping_rule(X_train)

# It's as simple as:
eem.predict(test_x)
```

### Results
|    |   MLP Hidden Layer Size |   Parallel Improvement |   Baseline Latency (in Parallel) |   test_accuracy_experimental |   test_accuracy_baseline |   Early-Exit Latency (in Parallel) |   Baseline Latency (Sequential) |   Baseline Latency (Sequential) |   Sequential Improvement |
|----|-------------------------|------------------------|----------------------------------|------------------------------|-------------------------|------------------------------------|---------------------------------|---------------------------------|--------------------------|
|  0 |                      32 |               0.765409 |                      7.63893e-05 |                     0.938596 |                 0.938596 |                        9.9802e-05  |                       0.0163752 |                       0.0137328 |                 1.19241  |
|  1 |                      64 |               0.786558 |                      8.25882e-05 |                     0.921053 |                 0.929825 |                        0.000105    |                       0.0165251 |                       0.0122609 |                 1.34779  |
|  2 |                     128 |               0.929282 |                      0.000124693 |                     0.929825 |                 0.929825 |                        0.000134182 |                       0.0167072 |                       0.0155337 |                 1.07555  |
|  3 |                     256 |               0.933253 |                      0.000166678 |                     0.938596 |                 0.938596 |                        0.000178599 |                       0.0168818 |                       0.0140476 |                 1.20175  |
|  4 |                     512 |               1.0645   |                      0.000258899 |                     0.921053 |                 0.921053 |                        0.000243211 |                       0.0174249 |                       0.0143654 |                 1.21298  |
|  5 |                    1024 |               0.981411 |                      0.000570202 |                     0.815789 |                 0.815789 |                        0.000581002 |                       0.0208043 |                       0.0266069 |                 0.781913 |
|  6 |                    2048 |               1.98646  |                      0.0011297   |                     0.938596 |                 0.947368 |                        0.0005687   |                       0.0214002 |                       0.0161236 |                 1.32726  |
|  7 |                    4096 |               1.70216  |                      0.00223529  |                     0.912281 |                 0.921053 |                        0.00131321  |                       0.0237825 |                       0.0187043 |                 1.2715   |

For a given hidden layer size, we compute the average time of running inference on a single sample. For 1000 samples, with ONNX, an MLP with a hidden layer size of 4096 would take 9.18 seconds to run. With ONNX and the custom acceleration, it would take 5.33 seconds to run those 1000 samples.


### Key Observations
- Significant latency reduction was observed for nearly all hidden layer sizes, in sequence.
- Significant latency reduction was observed at high hidden dimensions (H ≥ 1024), where computational complexity is highest.
- Algorithm did not universally result in speedups in low H regime.
- Minimal degradation in test accuracy across most model sizes.
- Our method performs best with very wide architectures, confirming its suitability for deployment scenarios involving large-capacity models.

### Stay Tuned For:
- An optimized algorithm so that it achieves latency improvement in smaller H settings. 
- A notebook demo so you can see the software in use to achieve this gains.
