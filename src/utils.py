import json
import os
from vars import *


def load_preds(f_name, method=1):
    PATH_F = os.path.join(PATH_D, f_name)
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


def max_score(score_list, i_max):  # method 3 for test
    s_index = i_max-4
    split_list = score_list[s_index:i_max]
    return max(split_list)
