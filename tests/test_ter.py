from nlgmetricverse import Nlgmetricverse
import os

scorer = Nlgmetricverse(metrics=["ter"])
predictions = os.getcwd() + "/predictions"
references = os.getcwd() + "/references"

scores = scorer(predictions=predictions, references=references, method="read_lines")
print("\nScores with method 'read_lines':")
print(scores)
print("\n---")

scores = scorer(predictions=predictions, references=references, method="no_new_line")
print("\nScores with method 'no_new_line':")
print(scores)
