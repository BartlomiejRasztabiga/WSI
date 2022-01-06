import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from svm import SVM


def read_data(file_name):
    x = [[] for _ in range(11)]
    y = []

    with open(file_name, newline='') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader, None)  # skip header
        for row in reader:
            for i in range(11):
                x[i].append(float(row[i]))
            wine_quality = float(row[11])
            y.append(1 if wine_quality > 5 else -1)

    return np.column_stack([np.array(x[i]) for i in range(11)]), y


def get_kernel(kernel_type: str, sigma=None):
    def linear_kernel(x1, x2):
        return np.dot(x1, x2.T)

    def rbf_kernel(x1, x2):
        return np.exp(-sigma * np.linalg.norm(x1 - x2) ** 2)

    if kernel_type == "linear":
        return linear_kernel
    else:
        return rbf_kernel


def run_svm(wine_feature_matrix, wine_qualities, kernel, c) -> tuple[float, np.ndarray]:
    model = SVM(kernel=kernel, c=c)

    x_train, x_test, y_train, y_test = train_test_split(wine_feature_matrix, wine_qualities, test_size=0.2,
                                                        random_state=2115)
    model.train(x_train, np.array(y_train))

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, confusion_matrix(y_test, y_pred)


def write_statistics_to_file(statistics):
    f = open('stats.json', "w")
    json.dump(statistics, f)
    f.close()


def read_statistics_from_file():
    f = open('stats.json', "r")
    statistics = json.load(f)
    f.close()
    return statistics


def generate_statistics():
    wine_feature_matrix, wine_qualities = read_data("winequality-red.csv")

    C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    S = [0.01, 0.1, 1, 10]

    statistics = []

    for c in C:
        accuracy, conf_matrix = run_svm(wine_feature_matrix, wine_qualities, get_kernel('linear'), c)
        statistics.append({"kernel": "linear", "c": c, "accuracy": accuracy, "confusion_matrix": conf_matrix.tolist()})
        for sigma in S:
            accuracy, conf_matrix = run_svm(wine_feature_matrix, wine_qualities, get_kernel('rbf', sigma=sigma), c)
            statistics.append(
                {"kernel": "rbf", "c": c, "sigma": sigma, "accuracy": accuracy,
                 "confusion_matrix": conf_matrix.tolist()})

    return statistics


def get_stats_entry(statistics, kernel, c, sigma):
    if kernel == 'linear':
        return list(filter(lambda stat: stat['kernel'] == kernel and stat['c'] == c, statistics))[0]
    else:
        return \
            list(filter(lambda stat: stat['kernel'] == kernel and stat['c'] == c and stat['sigma'] == sigma,
                        statistics))[0]


def draw_algorithms_comparison(statistics):
    fig, ax = plt.subplots()
    C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    for kernel in ["linear", "rbf"]:
        accuracies = []
        for c in C:
            stats = get_stats_entry(statistics, kernel, c, 0.1)
            accuracies.append(stats['accuracy'])
        ax.plot(C, accuracies, label=kernel)
    ax.legend()
    ax.set_xlabel('c')
    ax.set_ylabel('Dokładność klasyfikacji')
    ax.set_xscale('log')
    plt.grid()
    fig.savefig("comparison/kernel_comparison.png")
    plt.close(fig)


def draw_c_influence_on_linear_kernel(statistics):
    fig, ax = plt.subplots()
    C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    accuracies = []
    for c in C:
        stats = get_stats_entry(statistics, "linear", c, None)
        accuracies.append(stats['accuracy'])
    ax.plot(C, accuracies)
    plt.grid()
    ax.set_xlabel('c')
    ax.set_ylabel('Dokładność klasyfikacji')
    ax.set_xscale('log')
    plt.grid()
    fig.savefig("comparison/c_influence_linear.png")
    plt.close(fig)


def draw_c_influence_on_rbf_kernel(statistics):
    fig, ax = plt.subplots()
    C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    accuracies = []
    for c in C:
        stats = get_stats_entry(statistics, "rbf", c, 0.01)
        accuracies.append(stats['accuracy'])
    ax.plot(C, accuracies)
    plt.grid()
    ax.set_xlabel('c')
    ax.set_ylabel('Dokładność klasyfikacji')
    ax.set_xscale('log')
    plt.grid()
    fig.savefig("comparison/c_influence_rbf.png")
    plt.close(fig)


def draw_sigma_influence_on_rbf_kernel(statistics):
    fig, ax = plt.subplots()
    S = [0.01, 0.1, 1, 10]
    accuracies = []
    for sigma in S:
        stats = get_stats_entry(statistics, "rbf", 1000, sigma)
        accuracies.append(stats['accuracy'])
    ax.plot(S, accuracies)
    plt.grid()
    ax.set_xlabel('σ')
    ax.set_ylabel('Dokładność klasyfikacji')
    ax.set_xscale('log')
    plt.grid()
    fig.savefig("comparison/sigma_influence_rbf.png")
    plt.close(fig)


def generate_comparison(statistics):
    Path("./comparison").mkdir(exist_ok=True)
    draw_algorithms_comparison(statistics)
    draw_c_influence_on_linear_kernel(statistics)
    draw_c_influence_on_rbf_kernel(statistics)
    draw_sigma_influence_on_rbf_kernel(statistics)


def main():
    statistics = generate_statistics()
    # write_statistics_to_file(statistics)

    # statistics = read_statistics_from_file()
    generate_comparison(statistics)


if __name__ == "__main__":
    main()
