# Metric card for Compression

## Metric description
The Compression metric calculates the documention-summary word ration: <br><img src="https://render.githubusercontent.com/render/math?math={\frac{|x|}{|y|}}"><br>

### Inputs
- **predictions** (istance of EvaluationInstance): An object containing the predicted text.
- **references** (istance of EvaluationInstance): An object containing the reference text.

### Outputs
- **compression**(`float` or `int`): Compression score. Minimum possible value is 0. Maximum possible value is |x|

### Results from popular papers

## Bounds
The `compression` score has a <img src="https://render.githubusercontent.com/render/math?math={[0,|x|]}"> range.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
references = ["Peace at home, peace in th world.", "The cat is playing on the mat."]
scorer = NLGMetricverse(metrics=load_metric("compression"))
scores = scorer(predictions=predictions, references=references)
print(scores)
{ 
  "compression": { 
    'score' : 0.95 
  } 
}
```

## Limitations and Bias

## Citation(s)

## Further References