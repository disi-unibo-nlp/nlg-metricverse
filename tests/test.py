from nlgmetricverse import Nlgmetricverse

with open("predictions/predictions.txt", "r") as file:
    predictions_txt = file.readlines()
print("Predictions:")
print(predictions_txt)

with open('references/references.txt', 'r') as file:
    references_txt = file.readlines()
print("References:")
print(references_txt)
print()

scorer = Nlgmetricverse()
'''
scores = scorer(predictions=predictions_str, references=references_str)
'''
scores = scorer(predictions='predictions/', references='references/', method="no_new_line")
print("\nScores:")
print(scores)
