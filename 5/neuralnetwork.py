from math import sqrt
from typing import List

import numpy as np

DEBUG = True


class NeuralNetwork:
    def __init__(self, sizes: List[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes

        input_layer_size = 1

        self.biases = [
            np.random.uniform(-1.0 / sqrt(input_layer_size), 1.0 / sqrt(input_layer_size), size=(y, 1)).astype(
                np.longfloat) for y in sizes[1:]]
        self.weights = [
            np.random.uniform(-1.0 / sqrt(input_layer_size), 1.0 / sqrt(input_layer_size), size=(x, y)).astype(
                np.longfloat) for x, y in zip(sizes[:-1], sizes[1:])]

    def predict(self, x: float) -> float:
        return self._forward_propagation(x)

    def _forward_propagation(self, x: float) -> float:
        result = x
        for biases, weights in zip(self.biases, self.weights):
            result = self._sigmoid(np.matmul(weights, result) + biases)
        return result[0][0]

    def _backward_propagation(self, train_x: float, train_y: float):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = np.array([train_x])
        activations = [train_x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self._sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self._mse_derivative(activations[-1], train_y) * self._sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot( activations[-2].transpose(), delta)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self._sigmoid_derivative(z)
            delta = np.dot(delta, self.weights[-l + 1].transpose()) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def train(self, train_inputs, train_outputs, test_inputs, test_outputs, epochs_count: int, learning_rate: float,
              batch_size: int):

        for epoch_index in range(epochs_count):
            self._run_sgd(train_inputs, train_outputs, learning_rate, batch_size)

            if DEBUG is True and epoch_index % 100 == 0:
                nn_outputs = np.array([self.predict(x) for x in test_inputs])
                avg_mse = self._mean_squared_error(nn_outputs, test_outputs).mean()
                print(f"Epoch: {epoch_index} (MSE {avg_mse})")

    def _run_sgd(self, train_inputs, train_outputs, learning_rate: float, batch_size: int):
        train_samples_indexes = np.arange(len(train_inputs), dtype=int)
        np.random.shuffle(train_samples_indexes)  # shuffle indexes to have random data in each epoch
        mini_batches_count = len(train_samples_indexes) // batch_size

        for batch_num in range(mini_batches_count):
            start_index = batch_size * batch_num
            end_index = start_index + batch_size

            self._run_mini_batch(train_inputs, train_outputs, train_samples_indexes[start_index:end_index],
                                 learning_rate)

    def _run_mini_batch(self, train_xs, train_ys, train_indexes, learning_rate: float):
        biases_gradient = [np.zeros(b.shape) for b in self.biases]
        weights_gradient = [np.zeros(w.shape) for w in self.weights]
        for idx in train_indexes:
            biases_delta, weights_delta = self._backward_propagation(train_xs[idx], train_ys[idx])
            biases_gradient = [nb + dnb for nb, dnb in zip(biases_gradient, biases_delta)]
            weights_gradient = [nw + dnw for nw, dnw in zip(weights_gradient, weights_delta)]

        learning_step = learning_rate / len(train_indexes)

        self.weights = [w - learning_step * nw for w, nw in zip(self.weights, weights_gradient)]
        self.biases = [b - learning_step * nb for b, nb in zip(self.biases, biases_gradient)]

    def _sigmoid(self, x):
        ex = np.exp(x, dtype=np.longfloat)
        return ex / (ex + 1)

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _mean_squared_error(self, x, target):
        return np.square(x - target)

    def _mse_derivative(self, x, target):
        return 2 * (x - target)
