# Metric Card for *Current Metric*
***Metric Card Instructions:*** *Copy this file into the relevant metric folder, then fill it out and save it as README.md. Feel free to take a look at existing metric cards if you'd like examples.*

## Metric description
*Describe concisely the metric. What does it measure? How it works? What is it used for?*

### Inputs
*List all input arguments in the format below*
- **input1** (`type`): description of input1.

### Outputs
*List all output arguments in the format below*
- **output1**: description of output1.

### Results from popular papers
*If present, list some known input-output examples from popular papers (remember to cite)*

## Bounds
*Describe the boundary of the metric. What is the minimum and maximum value it can take?*

## Examples
*Provide examples of how to use the metric, and what the output looks like. You can use the code snippets below as a template.*
```python
from nlgmetricverse import NLGmetricverse, load_metric
scorer = NLGmetricverse(metrics=load_metric("wer"))
predictions = ["hello world", "good night moon"]
references = ["hello world", "good night moon"]
scores = scorer(predictions=predictions, references=references)
print(scores)
{
    "wer": {
        "score": 0.0
    }
}
```

## Limitations and bias
*Describe any known limitations or bias of the metric. For example, is it language-dependent? Does it only work for certain tasks?*

## Citations
*List any relevant citations here. If you use this metric in your work, please cite it as shown in the example below*
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
## References
*List any relevant references for documentation writing*
- [*name of paper*](link to paper)
