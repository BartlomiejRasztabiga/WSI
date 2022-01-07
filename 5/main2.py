import argparse
import sys

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from plot import plot


def f(x):
    return x ** 2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)


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
        # TODO refactor

        # input layer
        activations = np.array([[x]], dtype=np.longfloat)

        # hidden layer
        activations = self.forward_propagation(activations, self.hidden_layer_weights, self.hidden_layer_biases, True)

        # output layer
        activations = self.forward_propagation(activations, self.output_layer_weights, self.output_layer_biases, False)

        return activations[0][0]

    # Backwards propagation (learning)

    def train(self, train_inputs, train_outputs, test_inputs, test_outputs, epochs: int, mini_batch_size: int,
              learning_rate: float):
        # Train the network with SGD split into mini-batches

        train_indexes = np.arange(len(train_inputs), dtype=int)

        for i in range(epochs):
            self.run_sgd(train_inputs, train_outputs, train_indexes, mini_batch_size, learning_rate)

            if DEBUG is True and i % 100 == 0:
                avg_mse = mean_squared_error(np.array([self.predict(x) for x in test_inputs]), test_outputs).mean()
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
    SAMPLE_SIZE = 20000
    test_sample_multiplier = 0.1

    MIN_X = -15
    MAX_X = 15

    # Teach the network
    # TODO scaling only Y kind of works
    # scale_x = MinMaxScaler(feature_range=(-1, 1))
    scale_y = MinMaxScaler(feature_range=(-1, 1))

    # TODO optimise this!

    # train in/out
    train_inputs = np.linspace(MIN_X, MAX_X, SAMPLE_SIZE, dtype=np.longfloat)
    train_outputs = f(train_inputs)

    # reshape for scaler
    train_inputs = train_inputs.reshape((len(train_inputs), 1))
    train_outputs = train_outputs.reshape((len(train_outputs), 1))

    # scale train in/out
    # train_inputs = scale_x.fit_transform(train_inputs)
    train_outputs = scale_y.fit_transform(train_outputs)

    # test in/out
    test_inputs = np.linspace(MIN_X, MAX_X, int(SAMPLE_SIZE * test_sample_multiplier), dtype=np.longfloat)
    test_outputs = f(test_inputs)

    # reshape for scaler
    test_inputs = test_inputs.reshape((len(test_inputs), 1))
    test_outputs = test_outputs.reshape((len(test_outputs), 1))

    # scale test in/out
    # test_inputs = scale_x.fit_transform(test_inputs)
    test_outputs = scale_y.fit_transform(test_outputs)

    # reshape back for our neural net
    train_inputs = train_inputs.flatten()
    train_outputs = train_outputs.flatten()
    test_inputs = test_inputs.flatten()
    test_outputs = test_outputs.flatten()

    net = NeuralNetwork(hidden_layer_size)
    net.train(train_inputs, train_outputs, test_inputs, test_outputs, epochs, mini_batch_size, learning_rate)

    # gather predicted outputs
    nn_outputs = []
    for x in test_inputs:
        nn_outputs.append(net.predict(x))
    nn_outputs = np.array(nn_outputs)

    # reshape for reverse scaler
    test_inputs = test_inputs.reshape((len(test_inputs), 1))
    test_outputs = test_outputs.reshape((len(test_outputs), 1))
    nn_outputs = nn_outputs.reshape((len(nn_outputs), 1))

    # inverse scaling
    # test_inputs = scale_x.inverse_transform(test_inputs)
    test_outputs = scale_y.inverse_transform(test_outputs)
    nn_outputs = scale_y.inverse_transform(nn_outputs)

    # reshape back for csv results
    test_inputs = test_inputs.flatten()
    test_outputs = test_outputs.flatten()
    nn_outputs = nn_outputs.flatten()

    filename = f"range={MIN_X} n={hidden_layer_size} e={epochs} lr={learning_rate}.csv"

    with open(filename, "w") as file:
        for x, y, y_prediction in zip(test_inputs, test_outputs, nn_outputs):
            print(x, y, y_prediction, sep=",", file=file)

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
