"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        num_batches = n / mini_batch_size
        for j in range(epochs):
            random.shuffle(training_data)
            for k in range(0, num_batches):
                mini_batch = training_data[k * mini_batch_size: (k + 1) * mini_batch_size]
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def calculate_sum_derivatives_of_mini_batch(self, mini_batch):
        """
        计算m个样本的总梯度和。
        利用反向传播计算每一个样本(x,y)对应的梯度。
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # 给定一个样本X,利用反向传播算法计算对应w,b的梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 对m个样本的梯度进行累计求和
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        m = len(mini_batch)
        nabla_b, nabla_w = self.calculate_sum_derivatives_of_mini_batch(mini_batch)

        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights"."""
        # 初始化nb,nw,结构和b,w一样
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        # 执行算法的feedforward阶段
        # 　(1)初始化x作为a_1
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        # (2)l=2,....L层，分别计算z_l,a_l并且保存下来。
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # ========================================================================
        # 先计算所有的误差delta，最后计算所有层的梯度nb,nw，代码可读性更高一些
        # ========================================================================
        # method2
        # backward pass
        # 执行算法的backward阶段
        # (3)初始化第L层的误差,delta_L　= cost(a_L,y) * sigmoid_prime(z_L)
        l = -1
        delta = self.cost_derivative_of_a_L(activations[l], y) * sigmoid_prime(zs[l])
        deltas = [delta]  # list to store all the errors,layer by layer
        # (4)初始化l=L-1,....2层的误差,delta_l = np.dot(w_l+1^T,delta_l+1)* sigmoid_prime(z_l)
        for i in range(2, self.num_layers):
            l = -i  # (-２代表L-1,-3代表L-2,-(L-1)代表2)
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1]) * sigmoid_prime(zs[l])
            deltas.insert(0, delta)  # 确保误差的顺序，从后往前计算，所以需要insert在数组的最前面

        # (5)l=L,L-1,....2层，计算所有的梯度向量nb,nw
        for i in range(1, self.num_layers):
            l = -i  # (-1,-2,....-(L-1))
            nabla_b[l] = deltas[l]
            nabla_w[l] = np.dot(deltas[l], activations[l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        """
        l = [0,1,0,0,0,0,0,0,0,0]
        a = np.array(l).reshape(10,1)
        np.argmax(a) #输出向量对应的数字１

        test_results = [(1,1),(2,2),(3,3),(1,9)]
        [int(x == y) for (x, y) in test_results] 
        #[1, 1, 1, 0]
        sum([int(x == y) for (x, y) in test_results])
        #3
        """

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative_of_a_L(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
