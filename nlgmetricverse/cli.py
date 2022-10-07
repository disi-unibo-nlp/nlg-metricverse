"""
Utils CLI functions, for making simple, correct command line applications in Python.
"""
import glob
import json
import os.path
from typing import Any, Dict, List, Optional, Tuple, Union

import fire
import pandas as pd

from nlgmetricverse import NLGMetricverse
from nlgmetricverse import __version__ as nlgmetricverse_version
from nlgmetricverse.utils.sys import log
from nlgmetricverse.utils.data_structure import get_common_keys
from nlgmetricverse.utils.io import json_load, json_save


def file_extension(path: str) -> str:
    """
    Get file extension.

    :param path: File path.
    :return: File extension.
    """
    return ".".join(path.split("/")[-1].split(".")[1:])


def from_file(
    predictions: str,
    references: str,
    reduce_fn: Optional[str] = None,
    config: Optional[str] = "",
    export: Optional[str] = None,
):
    """
    Evaluate scores from file.

    :param predictions: Predictions.
    :param references: References.
    :param reduce_fn: Reduce function name.
    :param config: Optional configurations for some metrics.
    :param export: Optional, save scores in file.
    :return: Any.
    """
    args = json_load(config) or {}
    predictions = predictions or args.get("predictions")
    references = references or args.get("references")
    reduce_fn = reduce_fn or args.get("reduce_fn")
    metrics = args.get("metrics")
    scorer = NLGMetricverse(metrics=metrics)

    if os.path.isfile(predictions) and os.path.isfile(references):
        scores = score_from_file(scorer=scorer, predictions=predictions, references=references, reduce_fn=reduce_fn)
    elif os.path.isdir(predictions) and os.path.isdir(references):
        paths = read_folders(predictions, references)
        scores = {}
        for pred_file, ref_file in paths:
            common_name = os.path.basename(pred_file)
            scores[common_name] = score_from_file(
                scorer=scorer, predictions=pred_file, references=ref_file, reduce_fn=reduce_fn
            )
    else:
        raise ValueError("predictions and references either both must be files or both must be folders.")

    if export:
        json_save(scores, export)

    log(json.dumps(scores, default=str, indent=4))


def read_file(filepath: str) -> Union[List[str], List[List[str]]]:
    """
    Get content of a file.

    :param filepath: File path.
    :return: File content.
    """
    if file_extension(filepath) == "csv":
        df = pd.read_csv(filepath, header=None)
        content = df.to_numpy().tolist()
    elif file_extension(filepath) == "tsv":
        df = pd.read_csv(filepath, header=None)
        content = df.to_numpy().tolist()
    else:
        with open(filepath, "r") as in_file:
            content = in_file.readlines()
    return content


def read_folders(predictions_path: str, references_path: str) -> List[Tuple[str, str]]:
    """
    Get files from folders.

    :param predictions_path: Predictions path.
    :param references_path: References path.
    :return: Read files.
    """
    glob_predictions_path = os.path.join(predictions_path, "*")
    glob_references_path = os.path.join(references_path, "*")
    prediction_files = {os.path.basename(p): p for p in glob.glob(glob_predictions_path)}
    reference_files = {os.path.basename(p): p for p in glob.glob(glob_references_path)}

    common_files = get_common_keys(prediction_files, reference_files)

    files_to_read = []
    for common_file in common_files:
        common_pair = (prediction_files[common_file], reference_files[common_file])
        files_to_read.append(common_pair)

    return files_to_read


def score_from_file(scorer: NLGMetricverse, predictions: str, references: str, reduce_fn: Optional[str] = None) -> \
        Dict[str, Any]:
    """
    Get score from files.

    :param scorer: Main class application.
    :param predictions: Predictions.
    :param references: References.
    :param reduce_fn: Reduce function name.
    :return: scores
    """
    predictions = read_file(predictions)
    references = read_file(references)
    return scorer(predictions=predictions, references=references, reduce_fn=reduce_fn)


def app() -> None:
    """
    Cli app.
    """
    fire.Fire({"version": nlgmetricverse_version, "eval": from_file})


if __name__ == "__main__":
    app()
