from math import sin, cos
from random import random

import numpy as np


def f(x):
    return x ** 2 * sin(x) + 100 * sin(x) * cos(x)


class MLP(object):
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error):
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            # get activation for previous layer
            activations = self.activations[i + 1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0], -1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(inputs), i + 1))

        print("Training complete!")
        print("=====")

    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        return x * (1.0 - x)

    def _mse(self, target, output):
        """Mean Squared Error loss function
        Args:
            target (ndarray): The ground trut
            output (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)


if __name__ == "__main__":
    learning_set_multiplier = 0.8
    testing_set_multiplier = 1 - learning_set_multiplier

    # TODO hwo to do it better?
    X = np.array([[xi] for xi in np.arange(-40, 40, 0.1)])
    n = len(X)

    idx = np.random.choice(X.shape[0], size=int(n * learning_set_multiplier))
    X_train = X[idx]
    Y_train = np.array([[f(i[0])] for i in X_train])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(1, [40], 1)

    # train network
    X_train = np.array([[random()] for _ in range(1000)])
    Y_train = np.array([[f(i[0])] for i in X_train])
    mlp.train(X_train, Y_train, 100, 0.1)

    # create dummy data
    idx = np.random.choice(X.shape[0], size=int(n * testing_set_multiplier))
    X_test = X[idx]
    Y_test = np.array([[f(i[0])] for i in X_test])

    # create dummy data
    input = np.array([3])
    target = np.array([9])

    # get a prediction
    output = mlp.forward_propagate(input)

    print()
    print("Our network believes that f({}) is equal to {} and should be {}".format(input[0], output[0], f(input[0])))

    # TODO czemu z przykladem z neta dziala a tu nie? XD kwestia podawania danych?
    # TODO trzeba jakos normalizowac dane?
    # TODO zaleznie od funkcji aktywacji trzeba normalizowac dane, najlepiej chyba jakis tanh?
    # sigmoid zwraca wartosci tylko od -1 do 1!!!! chyba dlatego nie dziala? XD

    # get a prediction
    # for ix, x in enumerate(X_test):
    #     y = mlp.forward_propagate(x)
    #     print("Our network believesinputs that f({}) is equal to {}. Should be {}".format(x[0], y[0], Y_test[ix]))
