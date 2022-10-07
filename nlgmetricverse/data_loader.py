import glob
import os
from enum import Enum

from nlgmetricverse.metrics import EvaluationInstance


class DataLoaderStrategies(Enum):
    ReadLines = 0
    NoNewLine = 1


class DataLoader:
    """
    DataLoader for predictions and references. Those have to be paths to dirs.
    """

    def __init__(
            self,
            predictions,
            references,
            strategy=DataLoaderStrategies.NoNewLine,
    ):
        """
        :param predictions: Dir containing predictions.
        :param references: Dir containing references.
        :param strategy: Strategy to be applied for reading files.
        """
        self.res_predictions: EvaluationInstance = []
        self.res_references: EvaluationInstance = []
        self.dir_predictions = predictions
        self.dir_references = references
        self.execute_method(strategy)
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

    def execute_method(self, strategy):
        """
        Compute the strategy to be applied for scanning predictions and references. By default, is "no_new_line",
        but can also be "read_lines".

        :param strategy: Strategy to be applied.
        """
        if strategy == DataLoaderStrategies.ReadLines:
            self.read_lines(self.res_predictions, self.dir_predictions)
            self.read_lines(self.res_references, self.dir_references)
        elif strategy == DataLoaderStrategies.NoNewLine:
            self.no_new_line(self.res_predictions, self.dir_predictions)
            self.no_new_line(self.res_references, self.dir_references)
        return

    @staticmethod
    def read_lines(input_var, input_dir):
        """
        One element for each line.

        :param input_var: Var containing computes results.
        :param input_dir: Path to dir containing inputs.
        """
        cur_path = os.getcwd()
        os.chdir(input_dir)
        for f in sorted(glob.glob("*.txt")):
            with open(f, 'r', encoding="utf-8") as file:
                input_var.append(file.read().splitlines())
        os.chdir(cur_path)
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
            with open(f, 'r', encoding="utf-8") as file:
                input_var.append(file.read().replace('\n', ' '))
        return
