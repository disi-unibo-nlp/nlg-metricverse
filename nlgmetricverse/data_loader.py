import glob
import os
from enum import Enum

from nlgmetricverse.metrics import EvaluationInstance


class DataLoaderStrategies(Enum):
    OneRecordPerLine = 0
    OneRecordPerFile = 1


class DataLoader:
    """
    DataLoader for predictions and references. Those have to be paths to dirs.
    """

    def __init__(
            self,
            pred_path,
            ref_path,
            strategy=DataLoaderStrategies.OneRecordPerFile,
    ):
        """
        :param pred_path: Dir containing predictions.
        :param ref_path: Dir containing references.
        :param strategy: Strategy to be applied for reading files.
        """
        self.res_predictions: EvaluationInstance = []
        self.res_references: EvaluationInstance = []
        self.dir_predictions = pred_path
        self.dir_references = ref_path
        self.compute_strategy(strategy)
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

    def compute_strategy(self, strategy):
        """
        Compute the strategy to be applied for scanning predictions and references. By default, is "one_record_per_file",
        but can also be "one_record_per_line".

        :param strategy: Strategy to be applied.
        """
        if strategy == DataLoaderStrategies.OneRecordPerLine:
            self.one_record_per_line(self.res_predictions, self.dir_predictions)
            self.one_record_per_line(self.res_references, self.dir_references)
        elif strategy == DataLoaderStrategies.OneRecordPerFile:
            self.one_record_per_file(self.res_predictions, self.dir_predictions)
            self.one_record_per_file(self.res_references, self.dir_references)
        return

    @staticmethod
    def one_record_per_line(input_var, input_dir):
        """
        One record per line in file

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
    def one_record_per_file(input_var, input_dir):
        """
        One record per file in dir.

        :param input_var: Var containing computes results.
        :param input_dir: Path to dir containing inputs.
        """
        os.chdir(input_dir)
        for f in sorted(glob.glob("*.txt")):
            with open(f, 'r', encoding="utf-8") as file:
                input_var.append(file.read().replace('\n', ' '))
        return
