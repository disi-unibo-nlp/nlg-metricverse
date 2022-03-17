import glob
import os


class DataLoader:
    def __init__(
            self,
            predictions,
            references,
            method,
    ):
        self.res_predictions = None
        self.res_references = None
        self.check_inputs(predictions, references)
        method(self.res_predictions, predictions)
        method(self.res_references, references)

    @staticmethod
    def check_inputs(predictions, references):
        if predictions is None or references is None:
            raise TypeError("Both predictions and references have to be passed.")
        return

    def get_predictions(self):
        return self.res_predictions

    def get_references(self):
        return self.res_references

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
