import cvxopt
import numpy as np

# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False


class SVM(object):
    def __init__(self, kernel, c=1, minimum_lagrange_multiplier=1e-5):
        self.c = c
        self.kernel = kernel
        self.minimum_lagrange_multiplier = minimum_lagrange_multiplier

        self.lagrange_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.bias = None
        self.lambdas = None

    def train(self, x, y):
        n_samples, _ = np.shape(x)

        # Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(x[i], x[j])

        # P = (NxN) (kernel matrix)
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix)

        # q = -1 (1xN) [-1, -1, ...]
        q = cvxopt.matrix(-np.ones((n_samples, 1)))

        # λ constraints - between 0 and C
        G = cvxopt.matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.c)))

        # A = y^T [y1, y2, ..., ym]
        A = cvxopt.matrix(y, (1, n_samples), tc='d')

        # b = [0]
        b = cvxopt.matrix(0.0)

        # Solve the quadratic optimization problem using cvxopt
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        self.lagrange_multipliers = np.ravel(solution['x'])

        # Extract support vectors
        # Get indexes of non-zero Lagrange multipliers
        idx = self.lagrange_multipliers > self.minimum_lagrange_multiplier

        # Get the corresponding lagrange multipliers
        self.lambdas = self.lagrange_multipliers[idx]

        # Get the samples that will act as support vectors
        self.support_vectors = x[idx]

        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        # Calculate bias using first support vector
        self.bias = self.support_vector_labels[0]
        for i in range(len(self.lambdas)):
            self.bias -= self.lambdas[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i],
                                                                                       self.support_vectors[0])

    def predict(self, x):
        predicted_labels = []
        for sample in x:
            prediction = self.bias
            for i in range(len(self.lambdas)):
                prediction += self.lambdas[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i],
                                                                                            sample)
            predicted_labels.append(np.sign(prediction))
        return np.array(predicted_labels)
