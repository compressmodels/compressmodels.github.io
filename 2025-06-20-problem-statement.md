

## Problem Statement

### Inputs
1. Trained MLP model, $m$,  with $l$ hidden layers, each of hidden size $h_i$, where $0 \leq i < l$.
2. $X_{train} \in \mathbb{R}^{N_{train} \times D}$

### Define

$t(m, x)$ to be the time it takes model $m$ to process $x$.

### Objective

Minimize $\sum_i^N t(m_{new}, X_i)$ by designing $m_{new}$.

Achieving the following condition $\sum_{i}^{N} t(m_{new}, X_i) < \sum_i^N t(m, X_i)$ amounts to achieving an average latency speedup in inference.

The core insight to formulating an algorithm results from two critical hypotheses.

1. Some data is harder to classify than others.
2. The decision to classify *some* data *can be* easier than that of the classification itself.

With these two hypotheses in place, we create the following algorithm to solve this problem. 

At inference time, decide with $m_{decision}$ whether the data point is easy or hard. If it's easy, early exit and output a predicted class. If it's hard run the model as normal.

Define:
- Let $N_{easy}$ be the number of points classified as "easy" by $m_{decision}$.
- Let $N_{hard}$ be the number of points classified as "hard" by $m_{decision}$, so $N = N_{easy} + N_{hard}$.

Thus we can break the total time computation into the easy group and the hard group: 
$\sum_{i}^{N} t(m_{new}, X_i) = \sum_{i}^{N_{easy}} t(m_{decision}, X_i) + \sum_{i}^{N_{hard}} t(m, X_i) + t(m_{decision}, X_i)$.

Assuming $t(m, X_i)$ is constant for all $i$, we define $x$ to be an arbitrary $X_i$ this simplifies to:

= $N_{easy} t(m_{decision}, x) + N_{hard} (t(m, x) + t(m_{decision}, x))$

= $N_{easy} t(m_{decision}, x) + (N - N_{easy}) (t(m, x) + t(m_{decision}, x))$

To minimize this, we need to maximize $N_{easy}$, minimize $t(m_{decision}, x)$, all while maintaining accuracy.

#### Conditions for acceleration.

$$N_{easy}\ t(m_{decision}, x) + (N - N_{easy})\ (t(m, x) + t(m_{decision}, x)) < N\ t(m, x)$$

subject to all points within the easy group $m(x) = m_{decision}(x)$ to preserve accuracy.

Note that within the hard set, we need to evaluate the decision model, and only if it outputs hard do we need to evaluate the whole model.

## Choosing decision rules

A decision rule must be simple (low time complexity), maximize the number of points $N_{easy}$, and contain only one predicted class (so as to preserve accuracy)

There is a trade-off between these three concepts.
1. Choose a hypersphere around a single point yields a simple model with one class, but it suffers from the number of points (1) within it.
2. At another extreme, you can choose the entire dataset and have an extremely simple model, a large number of points, but low purity.
3. At the final extreme, you can overfit the model to be extremely high degree and include all the data points from one class and you have a large number of points, good purity but a highly complex and thus inefficient model.

### Methods

With the extremes of the problem stated for context, what we're doing is defining geometric rules (at the most simple (degree 1), hyperplanes, degree 2 conic sections) that achieve these objectives sufficiently for latency reduction. I use various methodologies to define these geometric rules. 

The algorithms I have implemented so far are:
- `add_linear_predict_rule`
- `add_hypersphere_prediction_grouping_rule`

### Results

See notebooks for empirical results.

