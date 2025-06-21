# moco

MLPs are common place in tabular models to predict stock market prices, click-through rates in advertising, as well as in the text domain based on their involvement in transformers.

On the edge, or situations where data is sufficiently high-volume, these models must operate fast or pay the price. In critical applications, this "price" can be huge, for autonomous vehicles it can be a crash, it can be lost revenue due to fraud, or failing to make a decision fast enough on a fleeting opportunity (in the case of algorithmic trading).

I am building practical methods that reduce the latency of multi-layer perceptrons (MLP) in inference as well as their energy-intensivity.


The software package has two functions available. Both functions construct early-exit rules that allow that model to branch given easy inputs.
- `add_linear_predict_rule`
- `add_hypersphere_prediction_grouping_rule` 

In both cases, the success (resulting time-savings) of the algorithm depends on the shape of the data.

Software Usage:
```python
from sklearn.neural_network import MLPClassifier
from moco import EarlyExitModel

model = MLPClassifier()
model.fit(X_train, y_train)

eem = EarlyExitModel(model)
eem.add_linear_predict_rule(X_train)

start = time.time()
eem.predict(X_test)
end = time.time()
experimental_time = end - start

start = time.time()
eem.baseline_predict(X_test)
end = time.time()
baseline_time = end - start

# Subject to the MLP being wide or deep!
assert baseline_time > experimental_time

```

Posts:
- [Problem Statement Blog Post](https://compressmodels.github.io/2025/06/20/problem-statement.html): Scientific Statement describing the problem I'm solving and how I'm solving it.
- [Credit Card Fraud Detection Blog Post](https://compressmodels.github.io/2025/06/06/realtime-fraud-detection.html): Achieving a 1.27x speed-up on wide and deep MLPs, sklearn, batched.
- [Breast Cancer Dataset Blog Post](https://compressmodels.github.io/2025/06/01/breast-cancer-case-study.html): Achieving a 2.4x speed-up on a wide MLP, `h = 2048` in-parallel, as well as >1.5 cost-savings across model sizes in the real-time setting.


We invite you to explore the detailed findings in our blog posts and consider integrating `moco` into your projects. Feel free to contact me at [quickmlmodels@gmail.com](mailto:quickmlmodels@gmail.com) if you have a model you want to try this out on.
