import pathlib

METRICS = [
    "bleu"
]
PATH_D = str(pathlib.Path(__file__).parent.parent.parent.resolve()) + '/data'
PATH_DEP = "requirements.txt"
data = {"id": "", "references": "", "predictions": "", "check": False}  # path to data sets
dep_solved = {}
