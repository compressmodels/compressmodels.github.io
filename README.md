# moco

AI is growing fast. LLMs are commonplace. 

There are two big problems that are receiving little corrective attention.
- LLMs are extremely energy-intensive $\rightarrow$ exacerbating climate change by increasing our dependence on fossil fuels.
- Models are a black box $\rightarrow$ we do not understand how they make decisions. This means they are high-risk inherently in safety-critical or otherwise high stakes scenarios.

My aim is to solve both problems, yielding interpretable and controllable models that have low energy costs.

I bring my background from graph theory, geometry, machine learning and software engineering to conduct practical, impactful research to solve these real-world problems.

I have started small and am building towards acheiving energy reductions for LLMs in inference. I have built from a first mathematical principles approach and have made traditional tabular models run faster (for fraud detection and medical use cases).

- [Credit Card Fraud Detection Blog Post](https://compressmodels.github.io/2025/06/06/realtime-fraud-detection.html): Achieving a 1.27x speed-up on wide and deep MLPs, sklearn, batched.
- [Breast Cancer Dataset Blog Post](https://compressmodels.github.io/2025/06/01/breast-cancer-case-study.html): Achieving a 2.4x speed-up on a wide MLP, `h = 2048` in-parallel, as well as >1.5 cost-savings across model sizes in the real-time setting.
- [Credit Card Fraud Detection XGBoost Case Study](https://compressmodels.github.io/research_report.pdf) Achieve ~-20% latency on XGBoost models.

Efficiency improvement in TinyBERT 
- [TinyBERT IMDB Case Study](https://compressmodels.github.io/tiny_bert_imdb.pdf) Achieve -21.5% improvement in efficient: translates to improvement for latency, throughput, energy usage with no accuracy tradeoff.


I invite you to explore the detailed findings in the blog posts and consider integrating `moco` into your projects. Feel free to contact me at [quickmlmodels@gmail.com](mailto:quickmlmodels@gmail.com) if you have a model you want to try this out on.
