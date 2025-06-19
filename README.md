# Project Page

I am working on research that makes multi-layer perceptrons (MLP) faster in inference (low latency) as well as less energy-intensive.

The software package has two functions available. Both functions construct early-exit rules that allow that model to branch given easy inputs.
- `add_linear_predict_rule` featured in the fraud dataset notebook example outputs a hyperplane (dividing the data space into two halfspaces). This hyperplane is fit such that all data in the + halfspace is one class. 
- `add_hypersphere_prediction_grouping_rule` featured in the breast cancer dataset notebook example uses various clustering techniques to detect groups with a high prevalence of a predicted class and then construct rules, defined as hyperspheres to detect membership within that group.

In both cases, the succcess of the algorithm will depend on the shape of the data.

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
- [Breast Cancer Dataset Blog Post](https://compressmodels.github.io/2025/06/01/breast-cancer-case-study.html): Achieving a 1.72x speed-up on a wide MLP, real-time and in-parallel.



Contact me at [quickmlmodels@gmail.com](mailto:quickmlmodels@gmail.com) if you want to get in touch.