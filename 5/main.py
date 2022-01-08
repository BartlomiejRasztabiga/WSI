import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from neuralnetwork import NeuralNetwork


def f(x):
    return x ** 2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)


def mse(x, target):
    return np.square(x - target)


def main(hidden_layer_size, epochs, mini_batch_size, learning_rate):
    min_x = -15
    max_x = 15
    range_length = abs(max_x - min_x)

    sample_size_per_range_unit = 1000
    sample_size = range_length * sample_size_per_range_unit

    test_sample_multiplier = 0.1

    # Teach the network
    # scaling just Y kind of works
    scale_y = MinMaxScaler(feature_range=(-1, 1))

    # train in/out
    train_inputs = np.linspace(min_x, max_x, sample_size, dtype=np.longfloat)
    train_outputs = f(train_inputs)

    # test in/out
    test_inputs = np.linspace(min_x, max_x, int(sample_size * test_sample_multiplier), dtype=np.longfloat)
    test_outputs = f(test_inputs)

    # scale train out
    train_outputs_scaled = train_outputs.reshape((len(train_outputs), 1))
    train_outputs_scaled = scale_y.fit_transform(train_outputs_scaled)
    train_outputs_scaled = train_outputs_scaled.flatten()

    # scale test out
    test_outputs_scaled = test_outputs.reshape((len(test_outputs), 1))
    test_outputs_scaled = scale_y.fit_transform(test_outputs_scaled)
    test_outputs_scaled = test_outputs_scaled.flatten()

    net = NeuralNetwork(hidden_layer_size)
    net.train(train_inputs, train_outputs_scaled, test_inputs, test_outputs_scaled, epochs, mini_batch_size,
              learning_rate)

    # gather predicted outputs
    nn_outputs = np.array([net.predict(x) for x in test_inputs])
    nn_outputs = nn_outputs.reshape((len(nn_outputs), 1))
    nn_outputs = scale_y.inverse_transform(nn_outputs)
    nn_outputs = nn_outputs.flatten()

    output_filename = f"range={min_x} n={hidden_layer_size} e={epochs} lr={learning_rate} batch={mini_batch_size}.csv"

    with open(output_filename, "w") as file:
        for x, y, y_prediction in zip(test_inputs, test_outputs, nn_outputs):
            print(x, y, y_prediction, sep=",", file=file)

    return output_filename


def plot_function(results_filename: str):
    xs = []
    ys_real = []
    ys_predicted = []

    results_file = open(results_filename, "r")
    lines = results_file.readlines()

    for line in lines:
        x, y_real, y_predicted = map(float, line.split(","))
        xs.append(x)
        ys_real.append(y_real)
        ys_predicted.append(y_predicted)

    avg_mse = mse(np.array(ys_predicted), np.array(ys_real)).mean()

    title = results_filename.replace(".csv", "").replace("./", "")

    plt.plot(xs, ys_real, "r")
    plt.plot(xs, ys_predicted, "b")
    plt.legend(["Real value", "Prediction"])
    plt.title(f"{title} MSE={round(avg_mse, 2)}")
    plt.grid(True)

    plt.savefig(f"{title}.png")
    plt.close()
    results_file.close()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        plot_function(sys.argv[1])
    else:
        hidden_layer_size = sys.argv[1]
        epochs = sys.argv[2]
        mini_batch_size = sys.argv[3]
        learning_rate = sys.argv[4]

        filename = main(hidden_layer_size, epochs, mini_batch_size, learning_rate)
        plot_function(filename)
