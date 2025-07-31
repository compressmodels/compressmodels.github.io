

# ⚡ moco
Low-Latency, Low-Energy AI Models — With No Accuracy Loss ⚡

![image](./images/graph.png)

## Use Cases


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

[Quantization](https://arxiv.org/abs/1712.05877), [pruning](https://arxiv.org/pdf/2308.06767), and knowledge distillation all involve achieving acceleration at the cost of accuracy.

Hardware acceleration with GPUs, faster CPUs or parallelizing ML models on many machines involve more money, infrastructure cost and energy cost.

Use `moco` to find subsets of data that are easy to classify early in the model, so the model does not need to execute fully for every single data point.

📢 Stay Tuned: I am building a software package. Indicate interest [here](https://forms.gle/TAYoxmpHGVZzrjiU6).

🚀 Try it on your model — <a href="mailto:quickmlmodels@gmail.com">Contact me</a>

## About Me


I use my background in graph theory and topological data analysis to develop algorithms built into software that data scientists and ML performance engineers can use to optimize latency-critical models.
