# Metric Card for chrF(++)

## Metric Description
ChrF and ChrF++ are two MT evaluation metrics.
They both use the F-score statistic for character n-gram matches, and ChrF++ adds word n-grams as well which correlates more strongly with direct assessment.

ChrF compares character n-grams in the reference and candidate sentences, instead of matching word n-grams as done in BLEU, ROUGE, etc.
The precision and recall are computed over the character n-grams for various values of n (upto 6) and are combined using arithmetic averaging to get the overall precision (chrP) and recall (chrR) respectively.
In other words, chrP represents the percentage of matched character n-grams present in the hypothesis and chrR represents the percentage of character n-grams in the reference which are also present in the hypothesis.
The final chrF score is then computed as:<br>
<img src="https://render.githubusercontent.com/render/math?math={chrF_{\beta} = (1 + \beta^2) \frac{chrP chrR}{\beta^2 chrP + chrR}}##gh-light-mode-only">,<br>
where <img src="https://render.githubusercontent.com/render/math?math={\beta}##gh-light-mode-only"> indicates that recall is given <img src="https://render.githubusercontent.com/render/math?math={\beta}##gh-light-mode-only"> times more weightage than precision.
Popovic propose two enhanced versions of chrF: (i) chrF+, which also considers word unigrams; (ii) chrF++, which considers word unigrams and bigrams in addition to character n-grams.

We use the implementation that is already present in sacreBLEU.
The implementation here is slightly different from sacreBLEU in terms of the required input format.
The length of the references and hypotheses lists need to be the same, so you may need to transpose your references compared to sacrebleu's required input format.
See [#3154 (comment)](https://github.com/huggingface/evaluate/issues/3154#issuecomment-950746534).

### Inputs
- **predictions** (`list`): prediction/candidate sentences.
- **references** (`list`): reference sentences.
- **char_order** (`int`): character n-gram order. Defaults to `6`.
- **word_order** (`int`): word n-gram order. If equals to 2, the metric is referred to as chrF++. Defaults to `0`.
- **beta** (`int`): determine the importance of recall w.r.t precision. Defaults to `2`.
- **lowercase** (`bool`): if `True`, enable case-insensitivity. Defaults to `False`.
- **whitespace** (`bool`): if `True`, include whitespaces when extracting character n-grams. Defaults to `False`.
- **eps_smoothing** (`bool`): if `True`, applies epsilon smoothing similar to reference chrF++.py, NLTK and Moses implementations. Otherwise, it takes into account effective match order similar to sacreBLEU < 2.0.0. Defaults to `False`.

### Outputs
ChrF(++) outputs a dictionary with the following values:
- **score** (`float`): the chrF(++) score.
- **char_order** (`int`): the selected character n-gram order.
- **word_order** (`int`): the selected word n-gram order.
- **beta** (`int`): the selected recall importance w.r.t. precision.

### Results from popular papers

## Bounds
The chrF(++) score can be any value in <img src="https://render.githubusercontent.com/render/math?math={[0,1]}##gh-light-mode-only">.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
scorer = NLGMetricverse(metrics=load_metric("chrf"))
predictions = [
  ["the cat is on the mat", "There is cat playing on the mat"],
  ["Look! what a wonderful day, today.", "Today is a very wonderful day"]
]
references = [
  ["the cat is playing on the mat.", "The cat plays on the mat."], 
  ["Today is a wonderful day", "The weather outside is wonderful."]
]
scores = scorer(predictions=predictions, references=references)
print(scores)
{
  "chrf": {
    'score': 0.44298405744188873, 
    'char_order': 6, 
    'word_order': 0, 
    'beta': 2
  }
}
```

## Limitations and bias
- According to [Popović 2017](https://www.statmt.org/wmt17/pdf/WMT70.pdf), chrF+ (where `word_order=1`) and chrF++ (where `word_order=2`) produce scores that correlate better with human judgements than chrF (where `word_order=0`) does. 

## Citation
```bibtex
@inproceedings{popovic-2015-chrf,
    title = "chr{F}: character n-gram {F}-score for automatic {MT} evaluation",
    author = "Popovi{\'c}, Maja",
    booktitle = "Proceedings of the Tenth Workshop on Statistical Machine Translation",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W15-3049",
    doi = "10.18653/v1/W15-3049",
    pages = "392--395",
}
@inproceedings{popovic-2017-chrf,
    title = "chr{F}++: words helping character n-grams",
    author = "Popovi{\'c}, Maja",
    booktitle = "Proceedings of the Second Conference on Machine Translation",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-4770",
    doi = "10.18653/v1/W17-4770",
    pages = "612--618",
}
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
```

## Further References
- See the [sacreBLEU README.md](https://github.com/mjpost/sacreBLEU#chrf--chrf) file for more information.