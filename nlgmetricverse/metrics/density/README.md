# Metric card for Density

## Metric description
The Density metric measures the average lenght of the extractive fragments. The density measure quantifies how well the word sequence of a summary can be described as a series of extractions. It is formulated as: $$ \frac{1}{|y|c}\sum{f \in \mathcal{F}(x,y)} |f|_c^2 $$
Where $||_c$ is the character lenght. When low, it suggest that most summary sentences are not verbatim extractions from the sources (abstractive).

### Inputs
- **predictions** (istance of EvaluationInstance): An object containing the predicted text.
- **references** (istance of EvaluationInstance): An object containing the reference text.

### Outputs
- **density**(`float` or `int`): Density score. Minimum possible value is 0. Maximum possible value is $|c|_c$. The lower is the score, the more abstractive the summary is.

### Results from popular papers

## Bounds
The `density` score has a $[0,|c|],\downarrow$ range.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["There is a cat on the mat.", "Look! a wonderful day."]
references = ["The cat is playing on the mat.", "Today is a wonderful day"]
scorer = NLGMetricverse(metrics=load_metric("density"))
scores = scorer(predictions=predictions, references=references)
print(scores)
{ 
  "density": { 
    'score' : 1.97
  } 
}
```

## Limitations and Bias

## Citation
```bibtex
@article{grusky2018newsroom,
  title={Newsroom: A dataset of 1.3 million summaries with diverse extractive strategies},
  author={Grusky, Max and Naaman, Mor and Artzi, Yoav},
  journal={arXiv preprint arXiv:1804.11283},
  year={2018}
}
```

## Further References

## Contributions
Thanks to @ValentinaPieri for contributing to this metric!