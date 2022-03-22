import os

from nlgmetricverse.repetitiveness import repetitiveness


predictions = os.getcwd() + "/predictions"
references = os.getcwd() + "/references"

rep = repetitiveness(predictions=predictions, references=references, method="no_new_line")
print("\nRepetitiveness:")
print(rep)
