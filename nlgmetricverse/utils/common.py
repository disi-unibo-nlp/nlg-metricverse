import os
import subprocess
import sys

from utils import vars


def init_dep_solved():
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])  # get all python packages
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    print(installed_packages)
    for metric in vars.METRICS:
        print(metric)
        vars.dep_solved.update({metric: []})  # set for all metrics python dependencies that must be installed to use it
        pth = "metrics/" + metric + "/" + vars.PATH_DEP
        dep_list = []
        if os.path.exists(pth):
            filed = open(pth, "r")
            content = filed.read()
            dep_list = content.split("\n")
            filed.close()
        removed_index = []
        if dep_list:
            for i in range(0, len(dep_list)):
                if dep_list[i] in installed_packages:
                    removed_index.append(i)  # filter list of dependency if already installed
            filtered_list = [j for i, j in enumerate(dep_list) if i not in removed_index]
            vars.dep_solved[metric] = filtered_list  # save the dependencies that must be installed


def load(metric):
    dependencies = vars.dep_solved[metric]
    if dependencies:  # if there's any dependencies when we load a metrics, install it!
        for dep in dependencies:
            script = "pip install " + dep
            os.system(script)
        vars.dep_solved[metric] = []


def load_preds(f_name, method=1):
    PATH_F = os.path.join(vars.PATH_D, f_name)
    with open(PATH_F, encoding="utf8") as f:
        lines = f.readlines()
    input_lines = s_format(lines)

    if method == 2:  # filtering list, take only the first input for each prediction (method 2 for test)
        f_list = []
        for n in range(0, len(input_lines), 4):
            f_list.append(input_lines[n])
        print(len(f_list))
        return f_list
    return input_lines


def s_format(lines=None):
    if lines is None:
        lines = []
    f_lines = []
    for line in lines:
        f_lines.append(line.translate({ord(i): None for i in "[,']"}))  # filter input lines
    return f_lines
