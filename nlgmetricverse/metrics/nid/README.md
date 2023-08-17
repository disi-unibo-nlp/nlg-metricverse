# Metric card for NID

## Metric description
The NID metric reckons redundancy by inverting the entropy of summary unigrams and applying lenght normalization: <br><img src="https://render.githubusercontent.com/render/math?math={1 - entropy(y)/log(|y|)}">

### Inputs
- **predictions** (istance of EvaluationInstance): An object containing the predicted text.
- **references** (istance of EvaluationInstance): An object containing the reference text.

### Outputs
- **nid**(`float` or `int`): NID score. Minimum possible value is 0. Maximum possible value is 1

### Results from popular papers

## Bounds
The `nid` score has a <img src="https://render.githubusercontent.com/render/math?math={[0,1]}"> range.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["There is a cat on the mat.", "Look! a wonderful day."]
references = ["The cat is playing on the mat.", "Today is a wonderful day"]
scorer = NLGMetricverse(metrics=load_metric("nid"))
scores = scorer(predictions=predictions, references=references)
print(scores)
{ 
  "nid": { 
    'score': 0.5101,
    }
}
```

## Limitations and Bias

## Citation

## Further References