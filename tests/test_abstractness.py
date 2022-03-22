import os

from nlgmetricverse.abstractness import abstractness


predictions = os.getcwd() + "/predictions"
references = os.getcwd() + "/references"

res = abstractness(predictions=predictions, references=references, method="read_lines")
print("Abstractness with method 'read_lines':")
print(res)
print("\n---")

res = abstractness(predictions=predictions, references=references, method="no_new_line")
print("Abstractness with method 'no_new_line':")
print(res)
