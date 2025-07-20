# moco

AI is growing fast. LLMs are commonplace. 

There are two big problems that are receiving little corrective attention.
- LLMs are extremely energy-intensive.
    - This exacerbates climate change by increasing our dependence on fossil fuels.
- Machine learning models are black boxes
    - We are scientifically limited in our ability to understand how they make decisions. This means they are high-risk inherently in safety-critical or otherwise high stakes scenarios.

### A network representing a transformer internal representation of the texts it classifies.
![Image](images/graph.png)
- red is negative label
- green is positive label


My aim is to solve both problems, yielding interpretable and controllable models that have low energy costs.

I bring my background from graph theory, geometry, machine learning and software engineering to conduct practical, impactful research to solve these real-world problems.

I have started small and am building towards acheiving energy reductions for LLMs in inference. I have built from a first mathematical principles approach and have made traditional tabular models run faster (for fraud detection and medical use cases).

- [Credit Card Fraud Detection Blog Post](https://compressmodels.github.io/2025/06/06/realtime-fraud-detection.html): Achieving a 1.27x speed-up on wide and deep MLPs, sklearn, batched.
- [Breast Cancer Dataset Blog Post](https://compressmodels.github.io/2025/06/01/breast-cancer-case-study.html): Achieving a 2.4x speed-up on a wide MLP, `h = 2048` in-parallel, as well as >1.5 cost-savings across model sizes in the real-time setting.
- [Credit Card Fraud Detection XGBoost Case Study](https://compressmodels.github.io/research_report.pdf) Achieve ~-20% latency on XGBoost models.

Recently, I've made the jump to transformers with promising results. 

Efficiency improvement in TinyBERT Classification
- [TinyBERT IMDB Case Study](https://compressmodels.github.io/tiny_bert_imdb.pdf) Achieve 21.5% reduction in latency & throughput on the IMDB movie review dataset, with no accuracy tradeoff, by early exiting.


I invite you to explore the detailed findings in the blog posts and consider integrating `moco` into your projects. Feel free to contact me at [quickmlmodels@gmail.com](mailto:quickmlmodels@gmail.com) if you have a model you want to try this out on. 

I'm looking for pilot opportunities. If you come across this and have a model that you need to hit a latency target, increase throughput or save energy costs, all without losing accuracy, let's talk. 
