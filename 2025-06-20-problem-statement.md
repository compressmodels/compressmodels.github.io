

## Problem Statement

Inputs:
- Trained MLP model, $m$,  with $l$ hidden layers, each of hidden size $h_i$, where $0 \leq i < l$.
- $X_{train} \in \mathbb{R}^{N_{train} \times D}$

Define: 
- $t(m, x)$ to be the time it takes model $m$ to process $x$.

Objective: 

Minimize $\sum_i^N t(m_{new}, X_i)$ by designing $m_{new}$.

To achieve model speed up $\sum_{i}^{N} t(m_{new}, X_i) < \sum_i^N t(m, X_i)$.

#

To achieve a result, there are two critical hypotheses.

1. Some data is harder to classify than others.
2. The decision to classify data as easy/hard is easier than that of the classification itself.

With these two hypotheses in place, we create the following algorithm to solve this problem: 

At inference time, decide whether the data point is easy or hard. If it's easy, early exit and output a predicted class. If it's hard run the model as normal.

Thus when we segment data by easy/hard status,

$\sum_{i}^{N} t(m_{new}, X_i) = \sum_{i}^{N_{easy}} t(m_{decision}, X_i) + \sum_{i}^{N_{hard}} t(m, X_i) + t(m_{decision}, X_i)$.

Assuming $t(m, X_i)$ is constant for all $i$, this simplifies to:

= $N_{easy} t(m_{decision}, X_i) + N_{hard} (t(m, X_i) + t(m_{decision}, X_i))$

where $N = N_{easy} + N_{hard}$.

= $N_{easy} t(m_{decision}, X_i) + (N - N_{easy}) (t(m, X_i) + t(m_{decision}, X_i))$

To minimize this, we need to maximize $N_{easy}$, minimize $t(m_{decision}, X_i)$, all while maintaining accuracy.

Note that within the hard set, we need to evaluate the decision model, and only if it outputs hard do we need to evaluate the whole model.

Now, at this point, the art is choosing decision rules that are simple (low time complexity), number of points (to achieve high bang for your buck) and contain only one predicted class (so as to preserve accuracy)

There is a trade-off between these three concepts. Choose a hypersphere around a single point yields a simple model with one class, but it suffers from the number of points (1) within it. At another extreme, you can choose the entire dataset and have an extremely simple model, a large number of points, but low purity. At the final extreme, you can overfit the model to be extremely high degree and include all the data points from one class and you have a large number of points, good purity but a highly complex and thus inefficient model.

##

### Methods

With the extremes of the problem stated for context, what we're doing is defining geometric rules (at the most simple (degree 1), hyperplanes, degree 2 conic sections) that achieve these objectives sufficiently for latency reduction. I use various methodologies to define these geometric rules. 

The algorithms I have implemented so far are:
- `add_linear_predict_rule`
- `add_hypersphere_prediction_grouping_rule`


### Results

See notebooks for empirical results.








