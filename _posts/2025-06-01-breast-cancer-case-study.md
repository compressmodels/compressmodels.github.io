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

We first note that, in terms of latency, ONNX outperforms in parallel, but lags behind sklearn significantly in real-time, one-at-a-time inference.

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

When we do this we observe minimal accuracy degradation.

|  | Hidden Layer Size | Accuracy (Baseline) |Accuracy (Experimental) |
|--|-------------------|---------------------|------------------------|
| 0|                32 |            0.938596 |               0.929825 |
| 1|                64 |            0.929825 |               0.929825 |
|2 |               128 |            0.929825 |               0.929825 |
| 3|             256   |            0.929825 |               0.929825 |
| 4|             512   |            0.921053 |               0.921053 |
| 5|            1024   |            0.921053 |               0.921053 |
| 6|            2048   |            0.921053 |               0.921053 |
| 7|            4096   |            0.921053 |               0.921053 |

Note how the accuracies are very nearly identical between the two runs. We are explicitly controlling that tradeoff.
#### Latency 🔥 

We show results for parallel latency  on average for the entire dataset as well as real-time average latency improvement in seconds for the entire dataset. This is computed as `baseline_time / experimental_time` so numbers greater than 1 mean time is being saved.

|    |   MLP Hidden Layer Size |   Parallel Improvement |   Sequential Improvement |
|---:|------------------------:|-----------------------:|-------------------------:|
|  0 |                      32 |               0.726136 |                  1.09331 |
|  1 |                      64 |               0.514278 |                  1.09343 |
|  2 |                     128 |               1.09549  |                  1.32097 |
|  3 |                     256 |               1.05646  |                  1.22204 |
|  4 |                     512 |               1.32636  |                  1.40272 |
|  5 |                    1024 |               1.82886  |                  1.32009 |
|  6 |                    2048 |               2.3913   |                  1.36833 |
|  7 |                    4096 |               1.45186  |                  1.32986 |


### Key Observations
- Significant latency reduction was observed for nearly all hidden layer sizes, in sequence.
- Significant latency reduction was observed at high hidden dimensions (H ≥ 1024), where computational complexity is highest.
- Our method performs best with very wide architectures, confirming its suitability for deployment scenarios involving large-capacity models.

### Next Steps
- Show this example with ONNX enabled.