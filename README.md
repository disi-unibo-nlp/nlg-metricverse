# nlg-eval 🖥️🧠 ➡️ 📜

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-blue?style=plastic&logo=python&logoColor=FFFB71)](#python)

## 🛠️ SETUP 🛠️

1. Import the project on a Python IDE (PyCharm recommended).
2. Run command on the terminal window: ``` pip install -r requirements.txt ```.
3. Create a new virtual environment or select your preferred python interpreter.

## 📋 UPDATE TEST SET 📋
1. Create '_src/data_' directory.
2. Put it file of references and predictions.
3. Run Python command: ``` update_test_set("name of test set", "references file name", "predictions file name") ```.
4. Results are saved in the metrics directory: '_name_of_test_set_name_metrics_'.

## ✔️ EXAMPLE OF USE ✔️
1. Move to the '_src_' directory.
2. Run Python commands:
```
>>> import blanche
>>> bl=blanche.Blanche()
>>> bl.update_test_set("t5_small", "preds_t5small_egv_beam.txt", "targets_t5small_egv_beam.txt")
>>> bl.run_rouge()
```
