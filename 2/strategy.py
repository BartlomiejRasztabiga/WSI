import math
from typing import Callable, Optional, List

import numpy as np
from numpy.random import Generator


class Individual:
    def __init__(self, x: np.ndarray, fitness: Optional[float], sigma: float):
        self.x = x
        self.fitness = fitness
        self.sigma = sigma

    def mutate(self, dimension: int, rng: Generator):
        mean = [0] * dimension
        cov = np.identity(dimension)
        self.x = self.x + self.sigma * rng.multivariate_normal(mean, cov)
        self.fitness = None

    def __str__(self):
        return "Individual(x={}, fitness={}, sigma={})".format(self.x, self.fitness, self.sigma)

    def __repr__(self):
        return self.__str__()


class EvolutionStrategy:
    def __init__(self, fitness_function: Callable, use_sigma_self_adaptation: bool, mi_population_size: int,
                 lambda_population_size: int, initial_sigma: float, initial_point: np.ndarray, max_iteration: int,
                 rng: Generator):
        self.fitness_function = fitness_function
        self.use_sigma_self_adaptation = use_sigma_self_adaptation
        self.mi_population_size = mi_population_size
        self.lambda_population_size = lambda_population_size
        self.initial_sigma = initial_sigma
        self.initial_point = initial_point
        self.max_iteration = max_iteration
        self.rng = rng

        self.dimension = len(initial_point)
        self.population = np.array([])
        self.best_individual = None
        self.avg_point = None
        self.function_values = []

    def generate_population(self):
        population = []
        avg_point_x = self.avg_point.x

        for _ in range(self.lambda_population_size):
            population.append(Individual(avg_point_x, None, self.avg_point.sigma))

        self.population = np.array(population)

    def update_sigma(self):
        self.sigma_self_adaptation() if self.use_sigma_self_adaptation else self.sigma_lmr_adaptation()

    def grade_population(self):
        for individual in self.population:
            fitness = self.fitness_function(individual.x)
            individual.fitness = fitness

    def sort_population_by_fitness(self) -> np.ndarray:
        return np.array(sorted(self.population, key=lambda individual: individual.fitness))

    def recombination(self):
        all_x = []
        all_sigmas = []

        for individual in self.population:
            all_x.append(individual.x)
            all_sigmas.append(individual.sigma)

        average_x = np.mean(all_x, axis=0)
        average_sigma = np.average(all_sigmas)
        fitness = self.fitness_function(average_x)

        self.avg_point = Individual(average_x, fitness, average_sigma)

    def mutation(self):
        for individual in self.population:
            individual.mutate(self.dimension, self.rng)

    def succession(self):
        self.population = self.sort_population_by_fitness()[:self.mi_population_size]

    def calculate_tau(self):
        return 1 / (2 * math.sqrt(self.dimension))

    def sigma_lmr_adaptation(self):
        tau = self.calculate_tau()
        ksi = math.e ** (tau * self.rng.normal(loc=0, scale=1.0))
        new_sigma_value = self.population[0].sigma * ksi

        for individual in self.population:
            individual.sigma = new_sigma_value

    def sigma_self_adaptation(self):
        tau = self.calculate_tau()

        for individual in self.population:
            individual.sigma *= math.e ** (tau * self.rng.normal(loc=0, scale=1.0))

    def run(self) -> tuple[Individual, int, List[float]]:
        iteration = 0

        self.avg_point = Individual(self.initial_point, None, self.initial_sigma)
        self.best_individual = self.avg_point

        while iteration < self.max_iteration:
            iteration += 1

            self.generate_population()
            self.update_sigma()
            self.mutation()
            self.grade_population()
            self.succession()
            self.recombination()

            self.function_values.append(self.fitness_function(self.avg_point.x))

            if self.avg_point.fitness < self.fitness_function(self.best_individual.x):
                self.best_individual = self.avg_point

        return self.best_individual, iteration, self.function_values
