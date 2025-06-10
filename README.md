# Project Page

I am working on research that makes multi-layer perceptrons (MLP) faster in inference (low latency) as well as less energy-intensive.

Posts:
- [Credit Card Fraud Detection Blog Post](https://compressmodels.github.io/2025/06/06/realtime-fraud-detection.html): Achieving a 1.27x speed-up on wide and deep MLPs, sklearn, batched.
- [Breast Cancer Dataset Blog Post](https://compressmodels.github.io/2025/06/01/breast-cancer-case-study.html): Achieving a 1.72x speed-up on a wide ONNX MLP, real-time.

How it works:
The software takes in a training dataset and a model and analyzes the datasets for groups of data points that are easy to classify using a combination of machine learning, optimization, geometric and topological techniques. Then, given those groups of data, the software constructs early-exit rules. At inference time, data points are evaluated for a given rule and early exited if the rule is satisfied. If this occurs sufficiently, it leads to large-scale average latency improvements.

Contact me at [quickmlmodels@gmail.com](mailto:quickmlmodels@gmail.com) if you want to get in touch.