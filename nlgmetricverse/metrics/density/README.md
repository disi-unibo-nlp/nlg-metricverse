# Metric card for Density

## Metric description
The Density metric measures the average lenght of the extractive fragments. it is formulated as: <br><img src="https://render.githubusercontent.com/render/math?math={\frac{1}{|y|} \sum_{f \in F(x,y)} (|f|_c)^2}"><br>
  
Where <img src="https://render.githubusercontent.com/render/math?math={||_c}"> is the character lenght. When low, it suggest that most summary sentences are not verbatim extractions from the sources (abstractive).

### Inputs
- **predictions** (istance of EvaluationInstance): An object containing the predicted text.
- **references** (istance of EvaluationInstance): An object containing the reference text.

### Outputs
- **density**(`float` or `int`): Density score. Minimum possible value is 0. Maximum possible value is <img src="https://render.githubusercontent.com/render/math?math={|x|_c}">

### Results from popular papers

## Bounds
The `density` score has a <img src="https://render.githubusercontent.com/render/math?math={[0,|x|_c]}"> range.

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

## Further References