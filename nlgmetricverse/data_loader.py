import glob
import os

from nlgmetricverse.metrics import EvaluationInstance


class DataLoader:
    """
    DataLoader for predictions and references. Those have to be paths to dirs.
    """
    def __init__(
            self,
            predictions,
            references,
            method,
    ):
        """
        :param predictions: Dir containing predictions.
        :param references: Dir containing references.
        :param method: Method to be applied.
        """
        self.res_predictions: EvaluationInstance = []
        self.res_references: EvaluationInstance = []
        self.dir_predictions = predictions
        self.dir_references = references
        self.execute_method(method)
        self.check_inputs()

    def check_inputs(self):
        """
        Check if references or predictions are empty.
        """
        if self.res_references is None or self.res_predictions is None:
            raise TypeError("Both predictions and references have to be passed.")
        return

    def get_predictions(self):
        """
        Return computed predictions.
        """
        return self.res_predictions

    def get_references(self):
        """
        Return computed references.
        """
        return self.res_references

    def execute_method(self, method):
        """
        Compute the method to be applied for scanning predictions and references. By default, is "no_new_line",
        but can also be "read_lines".

        :param method: Method to be applied.
        """
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

        :param input_var: Var containing computes results.
        :param input_dir: Path to dir containing inputs.
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

        :param input_var: Var containing computes results.
        :param input_dir: Path to dir containing inputs.
        """
        os.chdir(input_dir)
        for f in sorted(glob.glob("*.txt")):
            with open(f, 'r') as file:
                input_var += [file.read().replace('\n', ' ')]
        return
