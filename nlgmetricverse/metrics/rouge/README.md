# Metric Card for ROUGE

## Metric Description
ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing.
The metrics compare an automatically produced text against one or more (human-produced) references.
Note that ROUGE is case insensitive, meaning that upper case letters are treated the same way as lower case letters.
ROUGE metric variants are: ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S.
- ROUGE-N is similar to BLEU-N in counting the n-gram matches between the hypothesis and reference, however, it is recall-based (not precision-based).
- ROUGE-L measures the longest common subsequence (LCS) between a pair of sentences. ROUGE-L is a F-measure where the precision and recall are computed using the length of the LCS. Note that ROUGE-L does not check for consecutiveness of the matches as long as the word order is the same. It hence cannot differentiate between hypotheses that could have different semantic implications, as long as they have the same LCS even with different spatial positions of the words w.r.t the reference.<br>
  <img src="https://render.githubusercontent.com/render/math?math={P_{LCS}=\frac{|LCS|}{|words\_in\_hypothesis|}}"><br>
  <img src="https://render.githubusercontent.com/render/math?math={R_{LCS}=\frac{|LCS|}{|words\_in\_reference|}}"><br>
  <img src="https://render.githubusercontent.com/render/math?math={\text{ROUGE-L} = F_{LCS} = \frac{(1 + \beta^2)R_{LCS}P_{LCS}}{R_{LCS} + \beta^2 P_{LCS}}}">
- ROUGE-W addresses this by using a weighted LCS matching that adds a gap penalty to reduce weight on each non-consecutive match.
- ROUGE-S uses skip-bigram co-occurrence statistics to measure the similarity of the hypothesis and reference. Skipbigrams are pairs of words in the same sentence order, with arbitrary words in between. ROUGE-S is also computed as an F-score similar to ROUGE-L.

### Inputs
- **predictions** (`list`): list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
- **references** (`list`): list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
- **rouge_types** (`list`): list of rouge types to calculate. Defaults to `['rouge1', 'rouge2', 'rougeL', 'rougeLsum']`.
    - Valid rouge types:
        - `"rouge1"`: unigram (1-gram) based scoring
        - `"rouge2"`: bigram (2-gram) based scoring
        - `"rougeL"`: longest common subsequence based scoring
        - `"rougeLSum"`: splits text using `"\n"` (e.g., sentence/paragraph segments) and then calculates rougeL.
        - See [here](https://github.com/huggingface/datasets/issues/617) for more information
- **use_aggregator** (`boolean`): aggregation uses [bootstrap resampling](https://github.com/google-research/google-research/blob/master/rouge/scoring.py) to compute mid, high and low confidence intervals for precision, recall, and fmeasure as per the original ROUGE perl implementation. If `True`, aggregates the scores and returns the mid value for each rouge type. If `False` returns the selected metric(s) (`metric_to_select`) for each instance and rouge type. Defaults to `True`, forced to `True` in case of n-arity.
- **use_stemmer** (`boolean`): if `True`, uses Porter stemmer to strip word suffixes. Defaults to `False`.
- **metric_to_select** (`str`, optional): metric(s) to select between `precision`, `recall`, `fmeasure`. Defaults to `fmeasure`; if `None` returns all the three metrics.

**Note**: many papers says to use `rougeL` after a "\n"-splitting as a preprocessing step; this is equivalent to `rougeLsum` and generally produce higher scores.


### Outputs
ROUGE outputs a dictionary with one entry for each rouge type in the input list `rouge_types`. If `use_aggregator=False`, each dictionary entry is a list of float scores representing the selected metrics (see `metric_to_select`) for each rouge type.


## Bounds
The `precision`, `recall`, and `fmeasure` values all have a <img src="https://render.githubusercontent.com/render/math?math={[0,1]}"> range.


## Examples
```python
predictions = ["The quick brown fox jumped over the lazy dog."]
references = ["The fox jumped over the dog."]
scorer = Nlgmetricverse(metrics=load_metric("rouge"))
scores = scorer(predictions=predictions, references=references,
                rouge_types=["rougeL"],
                use_aggregator=False, use_stemmer=False,
                metric_to_select="fmeasure")
print(scores)
{ "total_items": 1, "empty_items": 0, "rouge": { "rougeL": 0.8 } }
```

## Limitations and Bias
See [Schluter (2017)](https://aclanthology.org/E17-2007/) for an in-depth discussion of many of ROUGE's limits.

## Citation
```bibtex
@inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W04-1013",
    pages = "74--81",
}
```

## Further References
- This metrics is a wrapper around the [Google Research reimplementation of ROUGE](https://github.com/google-research/google-research/tree/master/rouge)
- [To ROUGE or not to ROUGE](https://towardsdatascience.com/to-rouge-or-not-to-rouge-6a5f3552ea45)
- [ROUGE for summarization tasks](https://huggingface.co/course/chapter7/5?fw=tf)
