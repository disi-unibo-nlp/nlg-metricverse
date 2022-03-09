from nlgmetricverse import Nlgmetricverse

scorer = Nlgmetricverse()

predictions_str = [
    ["the cat is on the mat", "There is cat playing on the mat"],
    ["Look!    a wonderful day."]
]
references_str = [
    ["the cat is playing on the mat.", "The cat plays on the mat."],
    ["Today is a wonderful day", "The weather outside is wonderful."]
]

with open("predictions.txt", "r") as file:
    predictions_txt = file.readlines()

print(predictions_txt)

with open('references.txt', 'r') as file:
    references_txt = file.readlines()

print(references_txt)

scores = scorer(predictions=predictions_txt, references=references_txt)
print(scores)
