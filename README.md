<h1 align="center">nlg-metricverse 🌌</h1>

<table align="center">
    <tr>
        <td align="left">🚀 Spaceship</td>
        <td align="left">
          <a href="https://pypi.org/project/jury"><img src="https://img.shields.io/pypi/v/jury?color=blue" alt="PyPI"></a>
          <a href="https://pypi.org/project/jury"><img src="https://img.shields.io/pypi/pyversions/jury" alt="Python versions"></a>
          <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple&logo=python&logoColor=FFF800" alt="Made with Python"></a>
          <br>
          <a href="https://github.com/obss/jury/actions"><img alt="Build status" src="https://github.com/obss/jury/actions/workflows/ci.yml/badge.svg"></a>
          <a href="https://libraries.io/pypi/jury"><img alt="Dependencies" src="https://img.shields.io/librariesio/github/obss/jury"></a>
        </td>
    </tr>
    <tr>
        <td align="left">👨‍🚀 Astronauts</td>
        <td align="left">
          <a href="https://github.com/disi-unibo-nlp/nlg-metricverse/"><img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" alt="Open Source Love svg1"></a>
          <a href="https://github.com/obss/jury/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/pypi/l/jury"></a>
          <a href="https://GitHub.com/Nthakur20/StrapDown.js/graphs/commit-activity"><img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance"></a>
          <a href="https://pepy.tech/badge/beir"><img src="https://pepy.tech/badge/beir" alt="Downloads"></a>
        </td>
    </tr>
    <tr>
        <td align="left">🛰️ Training Program</td>
        <td align="left">
          <a href="https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
        </td>
    </tr>
    <tr>
        <td align="left">📕 Operating Manual</td>
        <td align="left">
          <a href="https://doi.org/10.5281/zenodo.6109838"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6109838.svg" alt="DOI"></a>
        </td>
    </tr>
</table>

<br>

> One NLG evaluation library to rule them all

<p align="center">
  <img src="./figures/nlgmetricverse_banner.png" title="nlg-metricverse" alt="">
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

## Installation

Install from PyPI repository
```
pip install nlg-metricverse
```
or build from source
```
git clone https://github.com/disi-unibo-nlp/nlg-metricverse.git
cd nlg-metricverse
pip install -v .
```
#### Explore on Hugging Face Spaces

The **Spaces** edition of NLG Metricverse launched on May X, 2022. Check it out here:
[![](./figures/spaces.png)](https://huggingface.co/spaces/nfel/Thermostat)

## Setup

## How to
