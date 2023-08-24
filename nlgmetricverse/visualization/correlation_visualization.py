import random
import matplotlib.pyplot as plt
import numpy as np


def mhc_visual(metrics, results):
    bar_list = plt.bar(metrics, results)

    for bar in bar_list:
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        bar.set_color(color)

    plt.xticks(np.arange(len(metrics)), metrics)
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Metric-human correlation")