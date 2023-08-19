# Metric card for Coverage

## Metric description
The Coverage metric measures the percentage of summary words within the source text: <br><img src="https://render.githubusercontent.com/render/math?math={\frac{1}{|y|} \sum_{f \in F(x,y)} |f|}##gh-light-mode-only"><br>

Where <img src="https://render.githubusercontent.com/render/math?math={F}##gh-light-mode-only"> is the set of all fragments, i.e., extractive character sequences. When low, it suggest a high change for unsupported entities and facts.

### Inputs
- **predictions** (istance of EvaluationInstance): An object containing the predicted text.
- **references** (istance of EvaluationInstance): An object containing the reference text.

### Outputs
- **coverage**(`float` or `int`): Coverage score. Minimum possible value is 0. Maximum possible value is 1. The higher the score, the higher percentage of summary words within the source text.

### Results from popular papers

## Bounds
The `coverage` score has a <img src="https://render.githubusercontent.com/render/math?math={[0,1]}"> range.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["There is a cat on the mat.", "Look! a wonderful day."]
references = ["The cat is playing on the mat.", "Today is a wonderful day"]
scorer = NLGMetricverse(metrics=load_metric("coverage"))
scores = scorer(predictions=predictions, references=references)
print(scores)
{ 
  "coverage": { 
    'score' : 0.77 
  } 
}
```

## Limitations and Bias

## Citation(s)

## Further References