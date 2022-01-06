import sys

import numpy as np
from matplotlib import pyplot as plt


def plot(filename: str):
    xs = np.array([], dtype=np.double)
    ys_real = np.array([], dtype=np.double)
    ys_predicted = np.array([], dtype=np.double)

    # TODO pamietac o odwroceniu normalizacji!

    file = open(filename, "r")
    lines = file.readlines()

    for line in lines:
        x, y_real, y_predicted = map(float, line.split(","))
        xs = np.append(xs, [x])
        ys_real = np.append(ys_real, [y_real])
        ys_predicted = np.append(ys_predicted, [y_predicted])

    plt.plot(xs, ys_real, "r")
    plt.plot(xs, ys_predicted, "b")
    plt.legend(["Actual", "Prediction"])
    plt.title(filename)
    # plt.xticks(np.arange(MIN_X, MAX_X, step=.5))
    # plt.yticks(np.arange(-2.2, 2.2, step=.2))
    plt.grid(True)

    title = filename.replace(".csv", "")
    plt.savefig(f"{title}.png")
    plt.close()
    file.close()


if __name__ == "__main__":
    plot(sys.argv[1])
