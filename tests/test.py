import os

from nlgmetricverse import Nlgmetricverse

scorer = Nlgmetricverse()
'''
scores = scorer(predictions=predictions_str, references=references_str)
'''
predictions = os.getcwd() + "/predictions"
references = os.getcwd() + "/references"
scores = scorer(predictions=predictions, references=references, method="read_lines")
print("\nScores:")
print(scores)
scores = scorer(predictions=predictions, references=references, method="no_new_line")
print("\nScores:")
print(scores)
