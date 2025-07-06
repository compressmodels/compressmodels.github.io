# moco

Machine learning models are common place in tabular models to predict stock market prices, and click-through rates in advertising.

On the edge, or situations where data is sufficiently high-volume, these models must operate fast. In critical applications, this "price" can be huge, for autonomous vehicles it can be a crash, it can be lost revenue due to fraud, or failing to make a decision fast enough on a fleeting opportunity (in the case of algorithmic trading).

I am building practical methods that reduce the latency and energy-intensivity of these prediction tasks in inference.

The inputs into the system are:
- the training data
- the model's predictions
- Optionally: the model itself or intermediate embeddings extracted from the model

The output:
- model-agnostic rules that are consistent with the model's predictions.

The success (resulting time-savings) of the algorithm depends on the data distribution.

I've demonstrated the software's success on multi-layer perceptrons (MLPs), but this approach is general enough to be applied to other model architectures.

The software package has two functions available. Both functions construct early-exit rules that allow the model to branch given easy inputs.
- `add_linear_definite_class_rule`
- `add_hypersphere_prediction_grouping_rule` 


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
- [Credit Card Fraud Detection Blog Post](https://compressmodels.github.io/2025/06/06/realtime-fraud-detection.html): Achieving a 1.27x speed-up on wide and deep MLPs, sklearn, batched.
- [Breast Cancer Dataset Blog Post](https://compressmodels.github.io/2025/06/01/breast-cancer-case-study.html): Achieving a 2.4x speed-up on a wide MLP, `h = 2048` in-parallel, as well as >1.5 cost-savings across model sizes in the real-time setting.


I invite you to explore the detailed findings in our blog posts and consider integrating `moco` into your projects. Feel free to contact me at [quickmlmodels@gmail.com](mailto:quickmlmodels@gmail.com) if you have a model you want to try this out on.
