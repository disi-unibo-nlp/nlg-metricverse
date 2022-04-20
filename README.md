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
          <a href="https://github.com/disi-unibo-nlp/nlg-metricverse/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/disi-unibo-nlp/nlg-metricverse.svg"></a>
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

## Tables Of Contents
- [Motivations](#-motivations)
- [Available Metrics](#-available-metrics)
    - [Hypothesis-Reference Supercluster](#hypothesis-reference-supercluster)
        - [N-gram Overlap Galaxy](#n-gram-overlap-galaxy)
        - [Embedding-based Galaxy](#embedding-based-galaxy)
    - [Hypothesis-only Supercluster](#hypothesis-only-supercluster)
- [Installation](#-installation)
    - [Explore on Hugging Face Spaces](#explore-on-hugging-face-spaces)
- [Quickstart](#-quickstart)
    - [Metric Selection](#metric-selection)
        - [Metric Documentation](#metric-documentation)
        - [Metric Filtering](#metric-filtering)
    - [Metric Usage](#metric-usage)
        - [Prediction-Reference Cardinality](#prediction-reference-cardinality)
        - [Scorer Application](#scorer-application)
        - [Metric-specific Parameters](#metric-specific-parameters)
        - [Outputs](#outputs)
- [Tests](#-tests)
    - [Code Style](#code-style)
- [Custom Metrics](#-custom-metrics)
- [Contributing](#-contributing)
- [Contact](#-contact)
- [License](#license)

## 💡 Motivations
* 📌 Human evaluation is often the best indicator of the quality of a system. However, designing crowd sourcing experiments is an expensive and high-latency process, which does not easily fit in a daily model development pipeline. Therefore, NLG researchers commonly use automatic evaluation metrics, which provide an acceptable proxy for quality and are very cheap to compute.
* 📌 NLG metrics aims to summarize and quantify the extent to which a model has managed to reproduce or accurately match some gold standard token sequences. Task examples: machine translation, abstractive question answering, single/multi-document summarization, data-to-text, chatbots, image/video captioning, etc.
* ☠ Different evaluation metrics encode **different properties** and have **different biases and other weaknesses**. Thus, you should choose your metrics carefully depending on your goals and motivate those choices when writing up and presenting your work.
* ☠ **As NLG models have gotten better over time, evaluation metrics have become an important bottleneck for research in this field**. In fact, areas can stagnate due to poor metrics, so we must be vigilant! It is an increasingly pressing priority to develop and apply better evaluation metrics given the recent advances in neural text generation. You shouldn't feel confined to the most traditional overlap-based metrics. If you're working on an established problem, you'll feel pressure from readers to be conservative and use the metrics that have already been tested for the same task. However, this might be a compelling pressure. Our view is that NLP engineers should enrich their evaluation toolkits with multiple metrics capturing different textual properties, being free to argue against cultural norms and motivate new ones, also exploring the latest contributions focused on semantics.
* ☠ NLG evaluation is very challenging also because **the relationships between candidate and reference texts tend to be one-to-many or many-to-many**. An artificial text predicted by a model might have multiple human references (i.e., there is more than one effective way to say most things), as well as a model can generate multiple distinct outputs. Such cardinality is crucial, but official implementations tend to neglect it.
* ☠ New NLG metrics are constantly being proposed in top conferences, but their **implementations (and related features) remain disrupted**, significantly restricting their application. Existing libraries tend to support a very small number of metrics, which mistakenly receive less attention than generative models. The absence of a shared and continuously updated repository makes it difficult to discover alternative metrics and slows down their use on a practical side.
* 🎯 NLG Metricverse implements a large number of prominent evaluation metrics in NLG, seeking to articulate the textual properties they encode (e.g., fluency, grammatical correctness, informativeness), tasks, and limits. Understanding, using, and examining a metric has never been easier.

## 🪐 Available Metrics
NLG Metricverse supports X diverse evaluation metrics overall (last update: May X, 2022).

### Hypothesis-Reference Supercluster

#### N-gram Overlap Galaxy

| Metric | Publication Year | Conference | NLG Metricverse | [Jury](https://github.com/obss/jury) | [HF/datasets](https://github.com/huggingface/datasets/tree/master/metrics) | [NLG-eval](https://github.com/Maluuba/nlg-eval) | [TorchMetrics](https://github.com/PyTorchLightning/metrics/tree/master/torchmetrics/text)
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| BLEU | 2002 | ACL | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| NIST | 2002 | HLT | :white_check_mark: | :x: | :x: | :x: | :x: |
| ORANGE (SentBLEU) | 2004 | COLING | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| ROUGE | 2004 | ACL | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| WER | 2004 | ICSLP | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| CER (TODO) | 2004 | ICSLP | | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| METEOR | 2005 | ACL | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: |
| CIDEr (TODO) | 2005 | | | :x: | :x: | :white_check_mark: | :x: |
| TER | 2006 | AMTA | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: |
| ChrF(++) | 2015 | ACL | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| CharacTER (TODO) | 2016 | WMT | | :x: | :x: | :x: | :x: |
| SacreBLEU | 2018 | ACL | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| METEOR++ (TODO) | 2018 | WMT | | :x: | :x: | :x: | :x: |
| Accuracy (TODO) | / | / | | :white_check_mark: | :white_check_mark: | :x: | :x: |
| Precision (TODO) | / | / | | :white_check_mark: | :white_check_mark: | :x: | :x: |
| F1 (TODO) | / | / | | :white_check_mark: | :white_check_mark: | :x: | :x: |
| MER (TODO) | / | / | | :x: | :x: | :x: | :white_check_mark: |

#### Embedding-based Galaxy

| Metric | Publication Year | Conference | NLG Metricverse | [Jury](https://github.com/obss/jury) | [HF/datasets](https://github.com/huggingface/datasets/tree/master/metrics) | [NLG-eval](https://github.com/Maluuba/nlg-eval) | [TorchMetrics](https://github.com/PyTorchLightning/metrics/tree/master/torchmetrics/text)
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| WMD (TODO) | 2015 | ICML | | :x: | :x: | :x: | :x: |
| SMD (TODO) | 2015 | ICML | | :x: | :x: | :x: | :x: |
| MOVERScore | 2019 | ACL | :white_check_mark: | :x: | :x: | :x: | :x: |
| EED (TODO) | 2019 | WMT | | :x: | :x: | :x: | :white_check_mark: |
| COMET | 2020 | EMNLP | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: |
| FactCC(X) (TODO) | 2020 | EMNLP | | :x: | :x: | :x: | :x: |
| BLEURT | 2020 | ACL | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: |
| NUBIA (TODO) | 2020 | EvalNLGEval<br>NeurIPS talk | | :x: | :x: | :x: | :x: |
| BERTScore | 2020 | ICLR | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| PRISM (TODO) | 2020 | EMNLP | | :white_check_mark: | :x: | :x: | :x: |
| BARTScore | 2021 | NeurIPS | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: |
| RoMe (TODO) | 2022 | ACL | | :x: | :x: | :x: | :x: |
| InfoLM (TODO) | 2022 | AAAI | | :x: | :x: | :x: | :x: |
| Perplexity (TODO) | / | / | | :x: | :white_check_mark: | :x: | :x: |
| Embedding Cosine Similarity (TODO) | / | / | | :x: | :x: | :white_check_mark: | :x: |
| Vector Extrema (TODO) | / | / | | :x: | :x: | :white_check_mark: | :x: |
| Greedy Matching (TODO) | / | / | | :x: | :x: | :white_check_mark: | :x: |
| Coverage (TODO) | ... | ... | | :x: | :x: | :x: | :x: |
| Density (TODO) | ... | ... | | :x: | :x: | :x: | :x: |
| Compression (TODO) | ... | ... | | :x: | :x: | :x: | :x: |

### Hypothesis-only Supercluster

| Metric | Publication Year | Conference | NLG Metricverse | [Jury](https://github.com/obss/jury) | [HF/datasets](https://github.com/huggingface/datasets/tree/master/metrics) | [NLG-eval](https://github.com/Maluuba/nlg-eval) | [TorchMetrics](https://github.com/PyTorchLightning/metrics/tree/master/torchmetrics/text)
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| MAUVE (TODO) | 2021 | NeurIPS | | :x: | :white_check_mark: | :x: | :x: |
| Flesch-Kincaid Readability | ... | ... | :white_check_mark: | :x: | :x: | :x: | :x: |
| Average Unique N-gram Ratios | ... | ... | :white_check_mark: | :x: | :x: | :x: | :x: |


## 🔌 Installation
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
[![](./figures/spaces.png)](https://huggingface.co/spaces/disi-unibo-nlp/nlg-metricverse)

## 🚀 Quickstart

It is only <b>two lines of code</b> to evaluate generated outputs: <b>(i)</b> instantiate your scorer by selecting the desired metric(s) and <b>(ii)</b> apply it!

### Metric Selection
Specify the metrics you want to use on instantiation,
```python
from nlgmetricverse import NLGMetricverse

# If you specify more metrics, each of them will be applyied on your data (allowing for a fast prediction/efficiency comparison)
scorer = NLGMetricverse(metrics=["bleu", "rouge"])
```
or directly import metrics from `nlgmetricverse.metrics` as classes, then instantiate and use them as desired.
```python
from nlgmetricverse.metrics import BertScore

scorer = BertScore.construct()
```
You can seemlessly access both `nlgmetricverse` and HuggingFace `datasets` metrics through `nlgmetricverse.load_metric`.
NLG Metricverse falls back to `datasets` implementation of metrics for the ones that are currently not supported; you can see the metrics available for `datasets` on [datasets/metrics](https://github.com/huggingface/datasets/tree/master/metrics). 
```python
import nlgmetricverse
bleu = nlgmetricverse.load_metric("bleu")
# metrics not available in `nlgmetricverse` but in `datasets`
wer = nlgmetricverse.load_metric("competition_math") # It falls back to `datasets` package with a warning
```
Note: if a selected metric requires specific packages, you'll be invited to install them (e.g., "bertscore" → `pip install bertscore`).

#### Metric Documentation
TODO

#### Metric Filtering
TODO

### Metric Usage

#### Prediction-Reference Cardinality
<i>1:1</i>. One prediction, one reference ([p<sub>1</sub>, ..., p<sub>n</sub>] and [r<sub>1</sub>, ..., r<sub>n</sub>] syntax).
```python
predictions = ["Evaluating artificial text has never been so simple", "the cat is on the mat"]
references = ["Evaluating artificial text is not difficult", "The cat is playing on the mat."]
```
<i>1:M</i>. One prediction, many references ([p<sub>1</sub>, ..., p<sub>n</sub>] and [[r<sub>11</sub>, ..., r<sub>1m</sub>], ..., [r<sub>n1</sub>, ..., r<sub>nm</sub>]] syntax)
```python
predictions = ["Evaluating artificial text has never been so simple", "the cat is on the mat"]
references = [
    ["Evaluating artificial text is not difficult", "Evaluating artificial text is simple"],
    ["The cat is playing on the mat.", "The cat plays on the mat."]
]
```
<i>K:M</i>. Many predictions, many references ([[p<sub>11</sub>, ..., p<sub>1k</sub>], ..., [p<sub>n1</sub>, ..., p<sub>nk</sub>]] and [[r<sub>11</sub>, ..., r<sub>1m</sub>], ..., [r<sub>n1</sub>, ..., r<sub>nm</sub>]] syntax). This is helpful for language models with a decoding strategy focused on diversity (e.g., beam search, temperature sampling).
```python
predictions = [
    ["Evaluating artificial text has never been so simple", "The evaluation of automatically generated text is simple."],
    ["the cat is on the mat", "the cat likes playing on the mat"]
]
references = [
    ["Evaluating artificial text is not difficult", "Evaluating artificial text is simple"],
    ["The cat is playing on the mat.", "The cat plays on the mat."]
]
```

#### Scorer Application
```python
scores = scorer(predictions, references)
```
The `scorer` automatically selects the proper strategy for applying the selected metric(s) depending on the input format. In any case, if a prediction needs to be compared against multiple references, you can customize the reduction function to use (e.g., `reduce_fn=max` chooses the prediction-reference pair with the highest score for each of the N items in the dataset).
```python
scores = scorer.compute(predictions, references, reduce_fn="max")
```

#### Metric-specific Parameters
Additional metric-specific parameters can be specified on `compute()`,
```python
# BertScore example for:
# - specifying which pre-trained BERT model use
# - asking to mount idf weighting
score = scorer.compute(predictions=predictions, references=references,
                       model_type="microsoft/deberta-large-mnli", idf=True)
```
or alternatively on instantiation.
```python
from nlgmetricverse.metrics import BertScore
scorer = BertScore.construct(compute_kwargs={
            "model_type": "microsoft/deberta-large-mnli",
            "idf": True})
score = scorer.compute(predictions=predictions, references=references)
```
```python
import nlgmetricverse
scorer = nlgmetricverse.load_metric(
            "bertscore",
            resulting_name="custom_bertscore",
            compute_kwargs={
                "model _type": "microsoft/deberta-large-mnli",
                "idf": True})
```

#### Outputs
TODO

## 🔎 Tests
TODO

### Code Style
To check the code style,
```
python tests/run_code_style.py check
```
To format the codebase,
```
python tests/run_code_style.py format
```

## 🎨 Custom Metrics
You can use custom metrics by inheriting `nlgmetricverse.metrics.Metric`.
You can see current metrics implemented on NLG Metricverse from [nlgmetricverse/metrics](https://github.com/disi-unibo-nlp/nlg-metricverse/tree/main/nlgmetricverse/metrics).
NLG Metricverse itself uses `datasets.Metric` as a base class to drive its own base class as `nlgmetricverse.metrics.Metric`. The interface is similar; however, NLG Metricverse makes the metrics to take a unified input type by handling metric-specific inputs and allowing multiple cardinalities (1:1, 1:M, K:M).
For implementing custom metrics, both base classes can be used but we strongly recommend using `nlgmetricverse.metrics.Metric` for its advantages.
```python
from nlgmetricverse.metrics import MetricForLanguageGeneration

class CustomMetric(MetricForLanguageGeneration):
    def _compute_single_pred_single_ref(
        self, predictions, references, reduce_fn = None, **kwargs
    ):
        raise NotImplementedError

    def _compute_single_pred_multi_ref(
        self, predictions, references, reduce_fn = None, **kwargs
    ):
        raise NotImplementedError

    def _compute_multi_pred_multi_ref(
            self, predictions, references, reduce_fn = None, **kwargs
    ):
        raise NotImplementedError
```
For more details, have a look at base metric implementation [nlgmetricverse.metrics.Metric](./nlgmetricverse/metrics/_core/base.py)

## 🙌 Contributing
Thanks go to all these wonderful collaborations for their contribution towards the NLG Metricverse library:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://giacomofrisoni.github.io/"><img src="https://github.com/giacomofrisoni.png" width="100px;" alt=""/><br /><sub><b>Giacomo Frisoni</b></sub></a></td>
    <td align="center"><a href="https://giacomofrisoni.github.io/"><img src="https://github.com/andreazammarchi3.png" width="100px;" alt=""/><br /><sub><b>Andrea Zammarchi</b></sub></a></td>
</table>

> We are hoping that the open-source community will help us edit the code and make it better!
> Don't hesitate to open issues and contribute the fix/improvement! We can guide you if you're not sure where to start but want to help us out 🥇.
> In order to contribute a change to our code base, please submit a pull request (PR) via GitHub and someone from our team will go over it and accept it.

> If you have troubles, suggestions, or ideas, the [Discussion](https://github.com/disi-unibo-nlp/nlg-metricverse/discussions) board might have some relevant information. If not, you can post your questions there 💬🗨.

## ✉ Contact
Contact person: Giacomo Frisoni, [giacomo.frisoni@unibo.it](mailto:giacomo.frisoni@unibo.it).

## License

The code is released under the [MIT License](LICENSE). It should not be used to promote or profit from violence, hate, and division, environmental destruction, abuse of human rights, or the destruction of people's physical and mental health.
