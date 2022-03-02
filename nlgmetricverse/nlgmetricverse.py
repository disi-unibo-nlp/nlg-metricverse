import os

from utils import common, vars
from metrics.bleu import run_bleu


class Nlgmetricverse:

    def __init__(self):
        vars.dep_solved = {}
        common.init_dep_solved()

    @staticmethod
    def run(metric, references=None, candidates=None):
        if candidates is None:
            candidates = []
        if references is None:
            references = []
        if metric == "bleu":
            run_bleu.run_bleu(references, candidates)

    @staticmethod
    def update_data(id_data, labels, predictions):
        if not os.path.exists(labels) and os.path.exists(predictions):
            print("One or both paths don't exist!")
            return False
        vars.data["id"] = id_data
        vars.data["references"] = vars.PATH_D + "/" + labels
        vars.data["predictions"] = vars.PATH_D + "/" + predictions
        vars.data["check"] = True

        print("Test set updated!")
        return True
