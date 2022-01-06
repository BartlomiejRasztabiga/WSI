import argparse
import sys

import numpy as np

from plot import plot

MIN_X = -15
MAX_X = 15


def f(x):
    return (x ** 2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)) / 100


EVAL_XS = np.linspace(MIN_X, MAX_X, 400, dtype=np.longfloat)
EVAL_YS = f(EVAL_XS)
DEBUG = True


def sigmoid(x):
    ex = np.exp(x, dtype=np.longfloat)
    return ex / (ex + 1)  # TODO check if correct


def sigmoid_derivative(x):
    return x * (1 - x)  # TODO check if correct


def mean_squared_error(x, target):
    return np.square(x - target)


def mse_derivative(x, target):
    return 2 * (x - target)


class NeuralNetwork:
    def __init__(self, hidden_layer_neurons_count: int = 9) -> None:
        self.hidden_layer_size = hidden_layer_neurons_count

        # TODO multi-layer

        self.hidden_layer_weights = np.random.uniform(-1.0, 1.0, size=(self.hidden_layer_size, 1)).astype(
            np.longfloat)  # TODO has to be uniform?
        self.hidden_layer_biases = np.random.uniform(-1.0, 1.0, size=(self.hidden_layer_size, 1)).astype(
            np.longfloat)  # TODO has to be uniform?

        self.output_layer_weights = np.zeros(shape=(1, self.hidden_layer_size), dtype=np.longfloat)
        self.output_layer_biases = np.zeros(shape=(1, 1))

    def forward_propagation(self, x, weights, biases, apply_activation: bool = True):
        out = np.matmul(weights, x) + biases
        return sigmoid(out) if apply_activation else out

    def predict(self, x: float) -> float:
        # input layer
        activations = np.array([[x]], dtype=np.longfloat)

        # hidden layer
        activations = self.forward_propagation(activations, self.hidden_layer_weights, self.hidden_layer_biases, True)

        # output layer
        activations = self.forward_propagation(activations, self.output_layer_weights, self.output_layer_biases, False)

        return activations[0][0]

    # Backwards propagation (learning)

    def train(self, train_inputs, train_outputs, epochs: int, mini_batch_size: int, learning_rate: float):
        # Train the network with SGD split into mini-batches

        train_indexes = np.arange(len(train_inputs), dtype=int)

        for i in range(epochs):
            self.run_sgd(train_inputs, train_outputs, train_indexes, mini_batch_size, learning_rate)

            if DEBUG is True and i % 100 == 0:
                avg_mse = mean_squared_error(np.array([self.predict(x) for x in EVAL_XS]), EVAL_YS).mean()
                print(f"Epoch: {i} (MSE {avg_mse})", file=sys.stderr)
        # print(min(losses))

    def run_sgd(self, train_inputs, train_outputs, train_indices, mini_batch_size: int, learning_rate: float):
        np.random.shuffle(train_indices)
        mini_batches_count = len(train_indices) // mini_batch_size

        for batch_num in range(mini_batches_count):
            start_index = mini_batch_size * batch_num
            end_index = start_index + mini_batch_size

            self.run_mini_batch(train_inputs, train_outputs, train_indices[start_index:end_index], learning_rate)

    def run_mini_batch(self, train_xs, train_ys, train_indexes, learning_rate: float):
        output_layer_weights_gradient = np.zeros(shape=self.output_layer_weights.shape, dtype=np.longfloat)
        hidden_layer_weights_gradient = np.zeros(shape=self.hidden_layer_weights.shape, dtype=np.longfloat)

        output_layer_biases_gradient = np.zeros(shape=self.output_layer_biases.shape, dtype=np.longfloat)
        hidden_layer_biases_gradient = np.zeros(shape=self.hidden_layer_biases.shape, dtype=np.longfloat)

        for idx in train_indexes:
            output_layer_weights_delta, hidden_layer_weights_delta, output_layer_bias_delta, hidden_layer_bias_delta = \
                self.backprop(train_xs[idx], train_ys[idx])

            output_layer_weights_gradient += output_layer_weights_delta
            hidden_layer_weights_gradient += hidden_layer_weights_delta

            output_layer_biases_gradient += output_layer_bias_delta
            hidden_layer_biases_gradient += hidden_layer_bias_delta

        learning_step = learning_rate / len(train_indexes)

        # TODO inline if possible
        output_layer_weights_gradient *= learning_step
        hidden_layer_weights_gradient *= learning_step

        output_layer_biases_gradient *= learning_step
        hidden_layer_biases_gradient *= learning_step

        self.output_layer_weights -= output_layer_weights_gradient
        self.hidden_layer_weights -= hidden_layer_weights_gradient

        self.output_layer_biases -= output_layer_biases_gradient
        self.hidden_layer_biases -= hidden_layer_biases_gradient

    def backprop(self, train_x: float, train_y: float):
        # Forward propagation
        output_layer_activations_expected = np.array([[train_y]], dtype=np.longfloat)
        input_layer_activations = np.array([[train_x]], dtype=np.longfloat)

        hidden_layer_activations = sigmoid(
            np.matmul(self.hidden_layer_weights, input_layer_activations) + self.hidden_layer_biases)

        output_layer_activations = np.matmul(self.output_layer_weights,
                                             hidden_layer_activations) + self.output_layer_biases

        # Backwards propagation into the output layer
        output_layer_bias_delta = mse_derivative(output_layer_activations, output_layer_activations_expected)
        output_layer_weights_delta = np.matmul(output_layer_bias_delta, hidden_layer_activations.T)

        # Backwards propagation into the hidden layer
        hidden_layer_bias_delta = np.matmul(self.output_layer_weights.T, output_layer_bias_delta)
        hidden_layer_bias_delta *= sigmoid_derivative(hidden_layer_activations)
        hidden_layer_weights_delta = np.matmul(hidden_layer_bias_delta, input_layer_activations)

        return output_layer_weights_delta, hidden_layer_weights_delta, output_layer_bias_delta, hidden_layer_bias_delta


def normalize(x, x_min, x_max, u=1, l=-1):
    return (x - x_min) * (u - l) / (x_max - x_min) + l


def main(hidden_layer_size, epochs, mini_batch_size, learning_rate):
    # Teach the network
    train_xs = np.linspace(MIN_X, MAX_X, 10000, dtype=np.longfloat)
    train_ys = f(train_xs)

    train_xs1 = []
    train_ys1 = []

    train_xs_min = min(train_xs)
    train_xs_max = max(train_xs)

    for xs in train_xs:
        train_xs1.append(normalize(xs, train_xs_min, train_xs_max))

    net = NeuralNetwork(hidden_layer_size)
    net.train(train_xs, train_ys, epochs, mini_batch_size, learning_rate)

    filename = f"range={MIN_X} n={hidden_layer_size} e={epochs} lr={learning_rate}.csv"

    with open(filename, "w") as file:
        for x, y in zip(EVAL_XS, EVAL_YS):
            print(x, y, net.predict(x), sep=",", file=file)

    return filename


# TODO SKALOWANIE!
# TODO znalezc dobre wartosci do -40, 40 albo to ojebac
# TODO po co mini batch, umiec wyjasnic, zmienic batch size?
# TODO po co learning rate, umiec wyjasnic, inna wartosc?
# TODO ile neuronow? tyle ile ekstremow?
# TODO po co bias, jak wplywa na to rozwiazanie, jak adaptowac
# TODO backprop na kartce
# TODO multi-layer? (chyba nie XD)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hidden_layer_size", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("mini_batch_size", type=int)
    parser.add_argument("learning_rate", type=float)
    args = parser.parse_args()

    filename = main(args.hidden_layer_size, args.epochs, args.mini_batch_size, args.learning_rate)
    plot(filename)
