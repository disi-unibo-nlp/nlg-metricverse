import glob
import os

from nlgmetricverse.metrics import EvaluationInstance


class DataLoader:
    def __init__(
            self,
            predictions,
            references,
            method,
    ):
        self.res_predictions: EvaluationInstance = []
        self.res_references: EvaluationInstance = []
        self.dir_predictions = predictions
        self.dir_references = references
        self.execute_method(method)
        self.check_inputs()

    def check_inputs(self):
        if self.res_references is None or self.res_references is None:
            raise TypeError("Both predictions and references have to be passed.")
        return

    def get_predictions(self):
        return self.res_predictions

    def get_references(self):
        return self.res_references

    def execute_method(self, method):
        if method == "read_lines":
            self.read_lines(self.res_predictions, self.dir_predictions)
            self.read_lines(self.res_references, self.dir_references)
        elif method == "no_new_line":
            self.no_new_line(self.res_predictions, self.dir_predictions)
            self.no_new_line(self.res_references, self.dir_references)
        print()
        print("Predictions: ")
        print(self.res_predictions)
        print("References: ")
        print(self.res_references)
        print()
        return

    @staticmethod
    def read_lines(input_var, input_dir):
        """
        One element for each line.
        """
        os.chdir(input_dir)
        for f in sorted(glob.glob("*.txt")):
            with open(f, 'r') as file:
                input_var += file.readlines()
        return

    @staticmethod
    def no_new_line(input_var, input_dir):
        """
        Multiline elements.
        """
        os.chdir(input_dir)
        for f in sorted(glob.glob("*.txt")):
            with open(f, 'r') as file:
                input_var += [file.read().replace('\n', ' ')]
        return
