# Project Page

I am working on research that makes multi-layer perceptrons (MLP) faster in inference (low latency) as well as less energy-intensive.

The software packages has two functions available for use by the user that result in latency improvements:
`add_linear_predict_rule` featured in the fraud dataset notebook example and `add_hypersphere_prediction_grouping_rule` featured in the breast cancer dataset notebook example.


Software Usage:
```[python]
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
- [Breast Cancer Dataset Blog Post](https://compressmodels.github.io/2025/06/01/breast-cancer-case-study.html): Achieving a 1.72x speed-up on a wide ONNX MLP, real-time.



Contact me at [quickmlmodels@gmail.com](mailto:quickmlmodels@gmail.com) if you want to get in touch.