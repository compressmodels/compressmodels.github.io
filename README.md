
`moco`: makes rate-limited and energy-limited classification ML models 15-30% more efficient.

## Use Cases
  - [Embedded Systems](https://compressmodels.github.io/one_pagers/moco-embedded-systems.pdf)
  - [Cybersecurity Network Intrusion](https://compressmodels.github.io/one_pagers/network_intrusion.pdf)
  - [Financial Fraud Detection](https://compressmodels.github.io/research_report.pdf)
  - [Sentiment Analysis](https://compressmodels.github.io/tiny_bert_imdb.pdf)

The basic Python usage (with an explanation) is the following:

```[python]
from sklearn.neural_network import MLPClassifier
from moco import LoggedFunction

# Train your model.
m = MLPClassifier()
m.fit(X, y)

# Wrap your predictor
lf = LoggedFunction(m.predict)

# Run your data through the wrapped model prediction function.
# Internally, we collect the input data as well as the output.
out = lf(X)

# We identify natural clusters in X that are entirely one class or another
# and construct rules based on these clusters.
rules = lf.optimize()

# Now when we run,
lf(X)

# the original model is only run when none of the rules indicate certainty in prediction.

```

## Benchmarked Results

| Dataset                                 |   Latency Improvement (Seq) |   Latency Improvement (Raced) |   Latency Improvement |   FLOPs Reduction |   Accuracy Change |
|:----------------------------------------|----------------------------:|------------------------------:|----------------------:|------------------:|------------------:|
| MNIST 8x8 Optical Character Recognition |                    50.8% |                      14.3% |              50.8% |          84.3% |      -0.0017  |
| Iris Flower Classification              |                    26.6% |                      22.5% |              26.6% |          66.9% |       0           |
| Credit Card Fraud Detection             |                    11.7% |                      58.5% |              58.5% |          31.9% |      -3.5e-06 |

| Dataset   |   Number of Datapoints |   Number of Features |   Hidden Layer |   Number of Classes |
|:----------|-----------------------:|---------------------:|---------------:|--------------------:|
| MNIST 8x8 Optical Character Recognition |                   1797 |                   64 |            100 |                  10 |
| Iris flower classification    |                    150 |                    4 |            100 |                   3 |
| Credit Card Fraud prediction   |                 284807 |                   28 |            100 |                   2 |


| Dataset                                 |   Latency (Baseline) |   Latency (Seq) |   Latency (Raced) |   Accuracy (Baseline) |   Accuracy (New) |   FLOPS (Baseline) |   FLOPS (New) |
|:----------------------------------------|---------------------:|----------------:|------------------:|----------------------:|-----------------:|-------------------:|--------------:|
| MNIST 8x8 Optical Character Recognition |          5.95014e-05 |     2.92691e-05 |       5.10171e-05 |              1        |         0.998331 |           13297800 |       2082480 |
|  1 | Iris Flower Classification              |          2.93296e-05 |     2.15416e-05 |       2.27195e-05 |              0.973333 |         0.973333 |             105000 |         34800 |
|  2 | Credit Card Fraud Detection             |          5.5073e-05  |     4.86502e-05 |       2.28601e-05 |              0.999709 |         0.999705 |          854421000 |     581988596 |

![image](./images/graph.png)


## Notebooks

### Natural Language Processing (NLP)
- [Accelerating TinyBERT Classification on IMDB dataset](https://compressmodels.github.io/tiny_bert_imdb.pdf) Achieve 21.5% reduction in latency & throughput on the IMDB movie review dataset with the transformer architecture, with no accuracy loss.

### Tabular

- [Credit Card Fraud Detection Latency improvement](https://compressmodels.github.io/2025/06/06/realtime-fraud-detection.html): Achieving a 1.27x speed-up on wide and deep MLPs, sklearn, batched.
- [Credit Card Fraud Detection XGBoost](https://compressmodels.github.io/research_report.pdf) Achieve ~-20% latency on XGBoost models.
- [Rapid Breast Cancer Diagnosis with sensor data](https://compressmodels.github.io/2025/06/01/breast-cancer-case-study.html): Achieving a 2.4x speed-up on a wide MLP, `h = 2048` in-parallel, as well as >1.5 cost-savings across model sizes in the real-time setting.

### Computer Vision (CV)
- [Rapid Handwritten Digit Classification](https://compressmodels.github.io/mnist_report.pdf)
Achieves a 20% latency improvement on the canonical MNIST dataset with a CNN model architecture.

## Achieving Faster Models

📢 Stay Tuned: I am building a software package. Indicate interest [here](https://forms.gle/TAYoxmpHGVZzrjiU6).

🚀 Try it on your model — <a href="mailto:quickmlmodels@gmail.com">Contact me</a>

View the project on GitHub at [moco-client](https://github.com/sam-randall/moco-client). PyPi package coming soon for you to try out.

## About Me

My name is Sam Randall. I attended Stanford University for my MS in Computational and Mathematical Engineering and my Bachelors in Applied Mathematics and Public Health from Johns Hopkins University. My research interests are in computational geometry, graph theory and their applications to machine learning.

I seek to make sustainability profitable and our world healthier so I am especially interested in partnering with organizations that are working to deploy models in very resource-constrained settings on edge devices to promote health and sustainability.

[LinkedIn](https://www.linkedin.com/in/sam-randall-9a3068110/)
