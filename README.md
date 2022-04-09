# nlg-metricverse 🌌

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-blue?style=plastic&logo=python&logoColor=FFF800)](#python)
---

<p align="center">
  <img src="./figs/nlgmetricverse_banner.png" title="nlg-metricverse" alt="">
</p>

### Explore the universe of Natural Language Generation (NLG) evaluation metrics.
NLG Metricverse is a living collection of NLG metrics in a unified and easy-to-use python environment.
* Reduces the implementational burden, allowing users to easily move from papers to practical applications.
* Increases comparability and replicability of NLG research.

## Motivations
* NLG metrics aims to summarize and quantify the extent to which a model has managed to reproduce or accurately match some gold standard token sequences. Task examples: machine translation, abstractive question answering, single/multi-document summarization, data-to-text, chatbots, image/video captioning, etc.
* Evaluation is very challenging also because the relationships tend to be _one-to-many_ or _many-to-many_. An artificial text predicted by a model might have multiple human references (i.e., there is more than one effective way to say most things), as well as a model can generate multiple distinct outputs.
* Different evaluation metrics encode _different properties_ and have _different biases and other weaknesses_. Thus, you should choose your metrics carefully depending on your goals and motivate those choices when writing up and presenting your work.
* You shouldn't feel confined to the most traditional metrics like ROUGE. If you're working on an established problem, you'll feel pressure from readers and referees to use the metrics that have already been tested for the problem. This might be a compelling pressure. Instead, you should always be free to argue against those cultural norms and motivate new ones, also exploring the latest contributions more focused on semantics. Areas can stagnate due to poor metrics, so we must be vigilant! However, just as new NLG metrics are constantly being proposed in top conferences, their implementations (and features) remain disrupted, restricting their application.
* NLG Metricverse implements a large number of prominent evaluation metrics in NLG, also seeking to articulate what textual properties they encode (e.g., fluency, grammatical correctness, informativeness), NLP applications, and their limits.

## Setup

## How to
