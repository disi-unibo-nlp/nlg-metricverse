# nlg-metricverse 🌌

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-blue?style=plastic&logo=python&logoColor=FFF800)](#python)
---

> One NLG evaluation library to rule them all

<p align="center">
  <img src="./figs/nlgmetricverse_banner.png" title="nlg-metricverse" alt="">
</p>

### Explore the universe of Natural Language Generation (NLG) evaluation metrics.
NLG Metricverse is a living collection of NLG metrics in a unified and easy-to-use python environment.
* Reduces the implementational burden, allowing users to easily move from papers to practical applications.
* Increases comparability and replicability of NLG research.

## Motivations
* 📌 NLG metrics aims to summarize and quantify the extent to which a model has managed to reproduce or accurately match some gold standard token sequences. Task examples: machine translation, abstractive question answering, single/multi-document summarization, data-to-text, chatbots, image/video captioning, etc.
* ☠ Different evaluation metrics encode **different properties** and have **different biases and other weaknesses**. Thus, you should choose your metrics carefully depending on your goals and motivate those choices when writing up and presenting your work.
* ☠ **As NLG models have gotten better over time, evaluation metrics have become an important bottleneck for research in this field**. In fact, areas can stagnate due to poor metrics, so we must be vigilant! It is an increasingly pressing priority to develop and apply better evaluation metrics given the recent advances in neural text generation. You shouldn't feel confined to the most traditional overlap-based metrics. If you're working on an established problem, you'll feel pressure from readers to be conservative and use the metrics that have already been tested for the same task. However, this might be a compelling pressure. Our view is that NLP engineers should enrich their evaluation toolkits with multiple metrics capturing different textual properties, being free to argue against cultural norms and motivate new ones, also exploring the latest contributions focused on semantics.
* ☠ NLG evaluation is very challenging also because **the relationships between candidate and reference texts tend to be one-to-many or many-to-many**. An artificial text predicted by a model might have multiple human references (i.e., there is more than one effective way to say most things), as well as a model can generate multiple distinct outputs. Such cardinality is crucial, but official implementations tend to neglect it.
* ☠ New NLG metrics are constantly being proposed in top conferences, but their **implementations (and related features) remain disrupted**, significantly restricting their application. Existing libraries tend to support a very small number of metrics, which mistakenly receive less attention than generative models. The absence of a shared and continuously updated repository makes it difficult to discover alternative metrics and slows down their use on a practical side.
* 🎯 NLG Metricverse implements a large number of prominent evaluation metrics in NLG, seeking to articulate the textual properties they encode (e.g., fluency, grammatical correctness, informativeness), tasks, and limits. Understanding, using, and examining a metric has never been easier.

## Setup

## How to
