---
layout: post
title: "Real Time Fraud Detection MLP"
date: 2025-06-06
author: "Sam Randall"
---

## Real Time Fraud Detection MLP Speedup

Accelerating Wide MLPs with Early Exit Inference: A Case Study on a Credit Card transaction dataset.

This blog post presents an empirical evaluation of a `sklearn` inference acceleration framework applied to wide multilayer perceptrons (MLPs), responsible for detecting fraud in credit card transactions. The method integrates early exit mechanisms and achieves a 1.27x speed-up, able to examine 1.27x more cases of fraud.

### Dataset
We evaluate our method on the [Credit Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) available on Kaggle, a benchmark for binary classification for fraud. The dataset contains 284807 data points and 28 features. We split it into train (80%), val 12% and test (12%).

### Experimental Setup

Model: `MLPClassifier` (scikit-learn)

Baseline: Standard sklearn model.

Accelerated: sklearn + early exit acceleration.

Metrics: Inference latency (s), training/test accuracy

Various Hidden Layer Architectures:
- One-layer MLP: an MLP with one hidden layer. We use values of 32, 64, 128 for the hidden size.
- Two-layer MLP: an MLP with two hidden layers. We use values of (32, 16), (32, 8), (64, 32), (64, 16), (128, 64), (128, 32) for the first and second layer size, respectively.

### Methods

#### Model Training

The exact balance of acceptable false positives versus acceptable false negatives is up to a business. Both missing a fraudulent transaction and incorrectly labeling a safe transaction as fraudulent are expensive, but in a system where manual review occurs if a model predicts fraud, missing a fraudulent transaction (a false negative) is far more expensive. One reasonable approach is to threshold recall at a threshold (for this example: 75%), and choose the model with the best precision, where that recall is exceeded. 

With this approach, we achieve the following precision and recall for our various model architectures (on a withheld test set):

|    | Architecture   |   Precision |   Recall |
|---:|:---------------|------------:|---------:|
|  0 | (16,)          |    0.875    | 0.660377 |
|  1 | (32,)          |    0.914286 | 0.603774 |
|  2 | (64,)          |    0.904762 | 0.716981 |
|  3 | (128,)         |    0.875    | 0.660377 |
|  4 | (256,)         |    0.875    | 0.660377 |
|  5 | (512,)         |    0.857143 | 0.679245 |
|  6 | (64, 32)       |    0.842105 | 0.603774 |
|  7 | (128, 64)      |    0.878049 | 0.679245 |
|  8 | (256, 128)     |    0.90625  | 0.54717  |

#### Improving Model Latency at Inference Time

##### Evaluation
For both the baseline and experimental approaches, we measure the total latency observed over the entire dataset.

We define adherence to be the proportion of data points that have the same predicted value across the baseline model and the experimental system. A value of 1 is perfect, a value of 0 means every prediction is wrong (worse than random).

#### Methods

The main aim is to locate obviously safe data, and classify those quickly.

We use the software package to find a linear classification rule to detect when a transaction is definitely safe. If this rule evaluates to `True`, then the system outputs "NOT FRAUD", and does not evaluate the original model. If the rule evaluates to `False`, indicating that the transaction is not "definitely safe", then we run the original model, and use its prediction as the system's output.

### Results

|    | MLP Architecture   |   sklearn MLP time (s) |   Gated MLP time (s) |   Speedup |   adherence |
|---:|:-------------------|-----------------------:|---------------------:|----------:|------------:|
|  0 | (16,)              |               0.016192 |             0.035199 |  0.460013 |           1 |
|  1 | (32,)              |               0.024527 |             0.038749 |  0.632971 |           1 |
|  2 | (64,)              |               0.040821 |             0.043826 |  0.931433 |           1 |
|  3 | (128,)             |               0.073607 |             0.073905 |  0.995968 |           1 |
|  4 | (256,)             |               0.159469 |             0.131458 |  1.21308  |           1 |
|  5 | (512,)             |               0.355406 |             0.277914 |  1.27883  |           1 |
|  6 | (64, 32)           |               0.066342 |             0.060172 |  1.10254  |           1 |
|  7 | (128, 64)          |               0.130396 |             0.110725 |  1.17766  |           1 |
|  8 | (256, 128)         |               0.30856  |             0.28904  |  1.06753  |           1 |

We averaged total latency over 5 runs of the dataset, and then rounded to six digits.

### Key Observations & Significance
- For this particular dataset, we see consistent improvements for wide and deep MLP architectures. 

If you're interested in seeing how this works on your MLP, get in touch at [quickmlmodels@gmail.com](quickmlmodels@gmail.com)

[Link to notebook](https://nbviewer.org/github/compressmodels/compressmodels.github.io/blob/main/notebooks/experiment-0606-fraud.ipynb)

