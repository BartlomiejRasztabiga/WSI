from math import sqrt

import numpy as np

DEBUG = True


class NeuralNetwork:
    def __init__(self, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size

        input_layer_size = 1
        output_layer_size = 1

        self.hidden_layer_weights = np.random.uniform(-1.0 / sqrt(input_layer_size), 1.0 / sqrt(input_layer_size),
                                                      size=(self.hidden_layer_size, input_layer_size)).astype(
            np.longfloat)
        self.hidden_layer_biases = np.random.uniform(-1.0 / sqrt(input_layer_size), 1.0 / sqrt(input_layer_size),
                                                     size=(self.hidden_layer_size, input_layer_size)).astype(
            np.longfloat)

        self.output_layer_weights = np.zeros(shape=(input_layer_size, self.hidden_layer_size), dtype=np.longfloat)
        self.output_layer_biases = np.zeros(shape=(output_layer_size, output_layer_size))

    def predict(self, x: float) -> float:
        return self._forward_propagation(x)

    def _forward_propagation(self, x: float) -> float:
        # input layer
        result = np.array([[x]], dtype=np.longfloat)

        # hidden layer
        result = self._sigmoid(np.matmul(self.hidden_layer_weights, result) + self.hidden_layer_biases)

        # output layer
        result = np.matmul(self.output_layer_weights, result) + self.output_layer_biases

        # output is a matrix 1x1
        return result[0][0]

    def _backward_propagation(self, train_x: float, train_y: float):
        # Forward propagation
        output_layer_expected_results = np.array([[train_y]], dtype=np.longfloat)
        input_layer_inputs = np.array([[train_x]], dtype=np.longfloat)

        hidden_layer_results = self._sigmoid(
            np.matmul(self.hidden_layer_weights, input_layer_inputs) + self.hidden_layer_biases)
        output_layer_results = np.matmul(self.output_layer_weights, hidden_layer_results) + self.output_layer_biases

        # Backward propagation into the output layer
        output_layer_bias_delta = self._mse_derivative(output_layer_results, output_layer_expected_results)
        output_layer_weights_delta = np.matmul(output_layer_bias_delta, hidden_layer_results.T)

        # Backward propagation into the hidden layer
        hidden_layer_bias_delta = np.matmul(self.output_layer_weights.T,
                                            output_layer_bias_delta) * self._sigmoid_derivative(hidden_layer_results)
        hidden_layer_weights_delta = np.matmul(hidden_layer_bias_delta, input_layer_inputs)

        return output_layer_weights_delta, hidden_layer_weights_delta, output_layer_bias_delta, hidden_layer_bias_delta

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
        output_layer_weights_gradient = np.zeros(shape=self.output_layer_weights.shape, dtype=np.longfloat)
        hidden_layer_weights_gradient = np.zeros(shape=self.hidden_layer_weights.shape, dtype=np.longfloat)
        output_layer_biases_gradient = np.zeros(shape=self.output_layer_biases.shape, dtype=np.longfloat)
        hidden_layer_biases_gradient = np.zeros(shape=self.hidden_layer_biases.shape, dtype=np.longfloat)

        for idx in train_indexes:
            output_layer_weights_delta, hidden_layer_weights_delta, output_layer_bias_delta, hidden_layer_bias_delta = \
                self._backward_propagation(train_xs[idx], train_ys[idx])

            output_layer_weights_gradient += output_layer_weights_delta
            hidden_layer_weights_gradient += hidden_layer_weights_delta
            output_layer_biases_gradient += output_layer_bias_delta
            hidden_layer_biases_gradient += hidden_layer_bias_delta

        learning_step = learning_rate / len(train_indexes)

        self.output_layer_weights -= output_layer_weights_gradient * learning_step
        self.hidden_layer_weights -= hidden_layer_weights_gradient * learning_step
        self.output_layer_biases -= output_layer_biases_gradient * learning_step
        self.hidden_layer_biases -= hidden_layer_biases_gradient * learning_step

    def _sigmoid(self, x):
        ex = np.exp(x, dtype=np.longfloat)
        return ex / (ex + 1)

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _mean_squared_error(self, x, target):
        return np.square(x - target)

    def _mse_derivative(self, x, target):
        return 2 * (x - target)
