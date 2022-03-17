from nlgmetricverse import Nlgmetricverse

'''
predictions_str = [
    ["the cat is on the mat", "There is cat playing on the mat"],
    ["Look!    a wonderful day."]
]
references_str = [
    ["the cat is playing on the mat.", "The cat plays on the mat."],
    ["Today is a wonderful day", "The weather outside is wonderful."]
]
'''
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
scores = scorer(predictions=predictions_txt, references=references_txt)
print("\nScores:")
print(scores)
