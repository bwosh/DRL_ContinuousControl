import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

def mean(data, length):
    cs = np.cumsum(data)
    mean100 = np.array((cs[length:]-cs[:-length])/length)
    trailing_zeros = np.repeat(np.NaN, length)
    return np.hstack([trailing_zeros, mean100])

def save_plot_results(title, values, mean_length, target_score, peek_length=15):
    plt.figure(figsize=(15,15))

    plt.hlines(target_score,0,len(values), linestyle='--', alpha=0.5)

    plt.plot(values, alpha=0.2, label="Train scores")
    plt.plot(mean(values, peek_length), alpha=0.5, label=f"Mean average over {peek_length} last episodes")
    plt.plot(mean(values, mean_length), label=F"Mean average over {mean_length} last episodes")

    plt.legend()
    plt.title(title)

    now = datetime.now()
    timedata = now.strftime("%Y%d%m_%H%M%S")
    filename = f"data/results_{title}_{timedata}.png"
    plt.savefig(filename)

