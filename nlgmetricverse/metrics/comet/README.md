# Metric Card for COMET

## Metric Description
Crosslingual Optimized Metric for Evaluation of Translation (COMET) is an open-source neural framework for generating multilingual-MT evaluation prediction estimates of three types of human judgments (HTER, DA's or MQM), training a model for each judgment type, achieving high-level correlations with the ground-truth scores and better robustness.
To encompass the distinct scoring types, the COMET framework supports two architectures with differnet training objectives:
(i) the Estimator model (targets = real values, i.e., HTER and MQM);
(ii) the Translation Ranking model (targets = relative rankings, i.e., DA).
While the Estimator is trained to regress directly on a quality score, the Translation Ranking model is trained to minimize the distance between a "better" hypothesis and both its corresponding reference and its original source.
Both models are composed of a pre-trained cross-lingual encoder (e.g., XLM-RoBERTa, multilingual BERT), and a pooling layer to produce sentence embeddings.
- The Estimator model independently encode the hypothesis and the reference (encoding), transforming the word embeddings into a sentence embedding for each segment (pooling).
Finally, the resulting sentence embeddings are combined and concatenated into one single vector that is passed to a feed-forward regressor.
The entire model is trained by minimizing the Mean Squared Error (MSE).
- The Translation Ranking model receives 4 segments: the source, the reference, a "better" hypothesis, and a "worse" one.
These segments are independently encoded using a pretrained cross-lingual encoder and a pooling layer on top.
Finally, using the triplet margin loss (Schroff et al., 2015), the resulting embedding space is optimized to minimize the distance between the "better" hypothesis and the "anchors" (source and reference).
With the release of the framework the authors also released fully trained models that were used to compete in the WMT20 Metrics Shared Task achieving SOTA in that years competition.

### Inputs
- **sources** (`list`): source sentences.
- **predictions** (`list`): candidate translations.
- **references** (`list`): reference translations.
- **gpus** (`int`): optional, an integer (number of GPUs to train on) or a list of integers (which GPUs to train on). Set to 0 to use CPU. The default value is None (uses one GPU if possible, else use CPU).
- **progress_bar** (`bool`): if set to True, progress updates will be printed out. The default value is `False`.
- **config_name** (`str`): COMET model to be used. Will default to `wmt20-comet-da` (previously known as `wmt-large-da-estimator-1719`) if None. Alternate models that can be chosen include `wmt20-comet-qe-da`, `wmt21-comet-mqm`, `wmt21-cometinho-da`, `wmt21-comet-qe-mqm` and `emnlp20-comet-rank`.

### Outputs
COMET outputs a dictionary with the following values:
- **samples** (`float`): the mean value of COMET scores `scores` over all the input sentences.
- **scores** (`list`): list of COMET scores for each of the input sentences.

### Results from popular papers
The [original COMET paper](https://arxiv.org/pdf/2009.09025.pdf) reported average COMET scores ranging from 0.4 to 0.6, depending on the language pairs used for evaluating translation models.
They also illustrate that COMET correlates well with human judgements compared to other metrics such as [BLEU](https://huggingface.co/metrics/bleu) and [CHRF](https://huggingface.co/metrics/chrf).

## Bounds
COMET scores approximately belong to <img src="https://render.githubusercontent.com/render/math?math={[0, 1]}##gh-light-mode-only">.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
scorer = NLGMetricverse(metrics=load_metric("comet"))
sources = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
predictions = ["The fire could be stopped", "Schools and kindergartens were open"]
references = ["They were able to control the fire", "Schools and kindergartens opened"]
scores = scorer(sources=sources, predictions=predictions, references=references)
print(scores)
{
   "comet": { 
      'scores': [
          0.1506408303976059, 
          0.915494441986084
      ], 
      'samples': 0.5330676361918449 
   } 
}
```

## Limitations and bias
The models provided for calculating the COMET metric are built on top of XLM-R and cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

Thus, results for language pairs containing uncovered languages are unreliable, as per the [COMET website](https://github.com/Unbabel/COMET)

Also, calculating the COMET metric involves downloading the model from which features are obtained -- the default model, `wmt20-comet-da`, takes over 1.79GB of storage space and downloading it can take a significant amount of time depending on the speed of your internet connection. If this is an issue, choose a smaller model; for instance `wmt21-cometinho-da` is 344MB.

## Citation
```bibtex
@inproceedings{rei-EtAl:2020:WMT,
   author    = {Rei, Ricardo  and  Stewart, Craig  and  Farinha, Ana C  and  Lavie, Alon},
   title     = {Unbabel's Participation in the WMT20 Metrics Shared Task},
   booktitle      = {Proceedings of the Fifth Conference on Machine Translation},
   month          = {November},
   year           = {2020},
   address        = {Online},
   publisher      = {Association for Computational Linguistics},
   pages     = {909--918},
}
@inproceedings{rei-etal-2020-comet,
   title = "{COMET}: A Neural Framework for {MT} Evaluation",
   author = "Rei, Ricardo  and
      Stewart, Craig  and
      Farinha, Ana C  and
      Lavie, Alon",
   booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
   month = nov,
   year = "2020",
   address = "Online",
   publisher = "Association for Computational Linguistics",
   url = "https://www.aclweb.org/anthology/2020.emnlp-main.213",
   pages = "2685--2702",
}
```

## Further References
- More information about model characteristics can be found on the [COMET official repository](https://unbabel.github.io/COMET/html/models.html).