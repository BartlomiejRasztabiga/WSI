from math import sin, cos

import matplotlib.pyplot as plt
import numpy as np

NUM_NEURONS = 4


def f(x):
    return x ** 2 * sin(x) + 100 * sin(x) * cos(x)


class NeuralNet:
    def __init__(self, x, y, lr):
        """Creating the structure of a neural network
        :param x: the input
        :param y: the output target
        """

        self.layer1 = None

        self.input = x

        self.lr = lr

        self.weight1 = np.random.rand(self.input.shape[1], NUM_NEURONS)
        self.weight2 = np.random.rand(NUM_NEURONS, 1)

        self.bias1 = np.zeros((self.input.shape[0], NUM_NEURONS))
        self.bias2 = np.zeros((self.input.shape[0], 1))

        self.y = y

        self.output = np.zeros((1, y.shape[1]))

    def feedforward(self):
        self.layer1 = self.__relu(np.dot(self.input, self.weight1) + self.bias1)
        self.output = self.__sigmoid(np.dot(self.layer1, self.weight2) + self.bias2)

    def backpropagation(self):
        d_z2 = self.output - y
        d_w2 = np.dot(self.layer1, d_z2)
        d_b2 = d_z2

        # Excepted shape (3, 4) for d_w1, actual (3, 3)

        d_z1 = np.dot(np.dot(self.weight2, d_z2.T), self.__relu_derivative(self.layer1))
        d_w1 = np.dot(self.input.T, d_z1)
        d_b1 = d_z1

        self.weight1 -= self.lr * d_w1
        self.weight2 -= self.lr * d_w2

        self.bias1 -= self.lr * d_b1
        self.bias2 -= self.lr * d_b2

    @staticmethod
    def calculate_cost(y, y_hat):
        num_train_examples = y_hat.shape[1]
        cost = np.sum(- np.dot(y_hat, np.log(y).T) - np.dot(1 - y_hat, np.log(1 - y).T)) / num_train_examples
        cost = np.squeeze(cost)

        return cost

    @staticmethod
    def __sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def __sigmoid_derivative(x):
        return x * (1.0 - x)

    @staticmethod
    def __relu(x):
        return np.maximum(0, x)

    @staticmethod
    def __relu_derivative(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


if __name__ == "__main__":

    def plot_cost(cost):
        plt.plot(cost.keys(), cost.values())

        plt.title("Training loss")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.grid(True)

        plt.xticks(list(range(0, 1500, 300)))  # Setting up the x ticks

        plt.show()


    x = np.array([[xi] for xi in np.arange(-40, 40, 0.1)]).T

    # X = np.array([[0],
    #               [0],
    #               [1],
    #               [1]])

    y = np.array([f(xi[0]) for xi in x]).T

    # y = np.array([[1],
    #               [0],
    #               [1],
    #               [0]])

    nn = NeuralNet(x, y, 0.0008)

    cost_reg = {}

    for i in range(800):
        np.random.seed(2000)
        nn.feedforward()
        nn.backpropagation()

        cost = nn.calculate_cost(nn.output, y)

        cost_reg[i] = cost

        print("After {} epochs, the cost is {}.".format(i, cost))

    print("\n")
    print(nn.output)
    plot_cost(cost_reg)
