# Metric card for Abstractness

## Metric description
The Abstractness metric measures how many new n-grams are present in the hypothesis compared to the references.
It is a Python module that provides a metric for measuring the abstractness of generated text. The metric computes the proportion of n-grams in the generated text that do not appear in the reference text. The abstractness score ranges from 0 to 1, with higher values indicating more abstract text.

A decrease in the abstractness of the sentence evaluated with respect to its reference was found by [recent studies](https://github.com/esdurmus/feqa): as the number of nodes in input to the model increases, the generated hypothesis becomes less and less abstract.

### Inputs
- **predictions** (istance of EvaluationInstance): An object containing the predicted text.
- **references** (istance of EvaluationInstance): An object containing the reference text.
- **n** (int): The size of the n-grams to use for computing abstractness.

### Outputs
- **abstractness**(`float` or `int`): Abstractness score. Minimum possible value is 0. Maximum possible value is 1.0. A higher score means higher abstractness.

### Results from popular papers

## Bounds
The `abstractness` values all have a <img src="https://render.githubusercontent.com/render/math?math={[0,1]}"> range.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["There is a cat on the mat.", "Look! a wonderful day."]
references = ["The cat is playing on the mat.", "Today is a wonderful day"]
scorer = NLGMetricverse(metrics=load_metric("abstractness"))
scores = scorer(predictions=predictions, references=references)
print(scores)
{ 
  "abstractness": { 
    'score' : 0.2727272727272727 
  } 
}
```

## Limitations and Bias

## Citation(s)
```bibtex
@inproceedings{durmus-etal-2020-feqa,
    title = "{FEQA}: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization",
    author = "Durmus, Esin  and
      He, He  and
      Diab, Mona",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.454",
    doi = "10.18653/v1/2020.acl-main.454",
    pages = "5055--5070",
    abstract = "Neural abstractive summarization models are prone to generate content inconsistent with the source document, i.e. unfaithful. Existing automatic metrics do not capture such mistakes effectively. We tackle the problem of evaluating faithfulness of a generated summary given its source document. We first collected human annotations of faithfulness for outputs from numerous models on two datasets. We find that current models exhibit a trade-off between abstractiveness and faithfulness: outputs with less word overlap with the source document are more likely to be unfaithful. Next, we propose an automatic question answering (QA) based metric for faithfulness, FEQA, which leverages recent advances in reading comprehension. Given question-answer pairs generated from the summary, a QA model extracts answers from the document; non-matched answers indicate unfaithful information in the summary. Among metrics based on word overlap, embedding similarity, and learned language understanding models, our QA-based metric has significantly higher correlation with human faithfulness scores, especially on highly abstractive summaries.",
}
```
## Further References