import random
import time

import numpy as np
import numdifftools as nd

from typing import Tuple, List

from numpy.linalg import LinAlgError


def timed(method):
    def _timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        return result, end - start

    return _timed


def get_initial_point(n):
    return [random.uniform(-100, 100) for _ in range(n)]


class Minimizer:
    def __init__(self, f, step=0.01, diff_tolerance=0.0001, max_number_of_iterations=50000,
                 initial_point=None):
        self.step = step
        self.diff_tolerance = diff_tolerance
        self.max_number_of_iterations = max_number_of_iterations
        self.f = f

        self.initial_point = initial_point

    def gradient(self):
        return nd.Gradient(self.f)

    def hessian(self):
        return nd.Hessian(self.f)

    @timed
    def simple_gradient_descent(self) -> Tuple[bool, List, List, int]:
        def find_new_step(_current_point, _difference, _step):
            beta = 0.8
            while self.f(_current_point - _step * _difference) >= self.f(current_point):
                _step *= beta
            return _step

        current_point = self.initial_point[:]
        gradient = self.gradient()
        iterations = 0
        function_values = []
        step = self.step
        for iterations in range(self.max_number_of_iterations + 1):
            function_values.append(self.f(current_point))
            gradient_at_point = gradient(current_point)

            difference = gradient_at_point
            step = find_new_step(current_point, difference, step)
            current_point += -step * difference

            if np.all(np.abs(difference) <= self.diff_tolerance):
                return True, current_point, function_values, iterations + 1

        return False, current_point, function_values, iterations + 1

    @timed
    def newton_descent(self) -> Tuple[bool, List, List, int]:
        current_point = self.initial_point[:]
        gradient = self.gradient()
        hessian = self.hessian()
        iterations = 0
        function_values = []
        for iterations in range(self.max_number_of_iterations + 1):
            function_values.append(self.f(current_point))
            gradient_at_point = gradient(current_point)
            hessian_at_point = hessian(current_point)
            inverted_hessian = np.linalg.inv(hessian_at_point)

            difference = np.matmul(inverted_hessian, gradient_at_point)
            current_point += -self.step * difference

            if np.all(np.abs(difference) <= self.diff_tolerance):
                return True, current_point, function_values, iterations + 1

        return False, current_point, function_values, iterations + 1

    @timed
    def newton_descent_adaptive_step(self) -> Tuple[bool, List, List, int]:
        def find_new_step(_current_point, _gradient_at_point, _difference, _step):
            t = 1.0
            alpha = 0.4
            beta = 0.8
            while self.f(_current_point - t * _step * _difference) > self.f(
                    _current_point) + alpha * t * _step * np.matmul(_gradient_at_point.T, -_difference):
                t *= beta
            return t * _step

        current_point = self.initial_point[:]
        gradient = self.gradient()
        hessian = self.hessian()
        iterations = 0
        function_values = []
        for iterations in range(self.max_number_of_iterations + 1):
            function_values.append(self.f(current_point))
            gradient_at_point = gradient(current_point)
            hessian_at_point = hessian(current_point)
            inverted_hessian = np.linalg.inv(hessian_at_point)

            difference = np.matmul(inverted_hessian, gradient_at_point)
            step = find_new_step(current_point, gradient_at_point, difference, self.step)
            current_point += -step * difference

            if np.all(np.abs(difference) <= self.diff_tolerance):
                return True, current_point, function_values, iterations + 1

        return False, current_point, function_values, iterations + 1
