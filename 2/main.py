import json
import random
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from strategy import EvolutionStrategy, Individual
from numpy.random import default_rng

np.set_printoptions(linewidth=np.inf)


def f(x: np.ndarray) -> float:
    assert len(x) == 10
    y = 0
    for xi in x:
        y += xi ** 2
    return y


def q(x: np.ndarray) -> float:
    norm = np.linalg.norm(x) ** 2

    result = 0
    result += (((norm - 10) ** 2) ** 0.125)
    result += 10 ** (-1) * (0.5 * norm + sum(x))
    result += 0.5
    return result


mi_population_sizes = [2, 10, 100]
initial_sigmas = [0.1, 1, 10]
fitness_functions = [f, q]


def get_avg_best_individual(best_individuals, fitness_function):
    all_x = []
    all_sigmas = []

    for individual in best_individuals:
        all_x.append(individual.x)
        all_sigmas.append(individual.sigma)

    average_x = np.mean(all_x, axis=0)
    average_sigma = np.average(all_sigmas)
    fitness = fitness_function(average_x)

    return Individual(average_x, fitness, average_sigma)


def run_evolution(fitness_function, use_sigma_self_adaptation, mi_population_size, lambda_population_size,
                  initial_sigma, initial_point, max_iteration, repetitions):
    best_individuals = []
    function_values = []
    seed = 21859

    for _ in range(repetitions):
        rng = default_rng(seed)
        strategy = EvolutionStrategy(fitness_function, use_sigma_self_adaptation, mi_population_size,
                                     lambda_population_size, initial_sigma, initial_point, max_iteration, rng)
        best_individual, iteration, values = strategy.run()
        # print(best_individual, iteration)
        best_individuals.append(best_individual)
        function_values.append(values)

        seed *= 2

    best_avg_individual = get_avg_best_individual(best_individuals, fitness_function)
    avg_values = np.mean(function_values, axis=0)
    return best_avg_individual, avg_values


def generate_statistics():
    repetitions = 10
    max_iteration = 1000
    dimensions = 10

    print("Running ES to get some statistics, repetitions={}, max_iteration={}".format(repetitions, max_iteration))
    statistics = []
    for use_sigma_self_adaptation in [False, True]:
        for mi_population_size in mi_population_sizes:
            for initial_sigma in initial_sigmas:
                for fitness_function in fitness_functions:
                    lambda_population_size = mi_population_size * 2
                    initial_point = np.array([random.uniform(-100, 100) for _ in range(dimensions)])

                    sigma_adaptation_algorithm_name = "SA" if use_sigma_self_adaptation else "LMR"

                    print("{:<1} sigma_adaptation={:<3} mi={:<4} initial_sigma={:<4}".format(fitness_function.__name__,
                                                                                             sigma_adaptation_algorithm_name,
                                                                                             mi_population_size,
                                                                                             initial_sigma), end=" - ")
                    best_individual, values = run_evolution(fitness_function, use_sigma_self_adaptation,
                                                            mi_population_size, lambda_population_size, initial_sigma,
                                                            initial_point, max_iteration, repetitions)
                    print(
                        "x={:<165} fitness={:<25} sigma={:<20}".format(str(best_individual.x), best_individual.fitness,
                                                                       best_individual.sigma))

                    stats = {"sigma_adaptation_algorithm": sigma_adaptation_algorithm_name,
                             "fitness_function": fitness_function.__name__, "mi": mi_population_size,
                             "initial_sigma": initial_sigma, "best_x": best_individual.x.tolist(),
                             "best_fitness": best_individual.fitness, "best_sigma": best_individual.sigma,
                             "function_values": values.tolist()}
                    statistics.append(stats)
    return statistics


def get_stats_entry(statistics, func, mi, sigma, sigma_adaptation):
    return list(filter(
        lambda stat: stat['fitness_function'] == func and stat['mi'] == mi and stat['initial_sigma'] == sigma and stat[
            'sigma_adaptation_algorithm'] == sigma_adaptation, statistics))[0]


def generate_algorithms_comparison_tex(statistics):
    for function in ["f", "q"]:
        lines = []
        for mi in mi_population_sizes:
            for sigma in initial_sigmas:
                lmr = get_stats_entry(statistics, function, mi, sigma, "LMR")
                sa = get_stats_entry(statistics, function, mi, sigma, "SA")
                lines.append("{} & {} & {:.3e} & {:.3e} \\\\\n\\hline".format(mi, sigma, lmr['best_fitness'],
                                                                              sa['best_fitness']))
            lines.append("\\hline")
        file = open("./comparison/{}.tex".format(function), "w")
        file.writelines(lines)
        file.close()


def draw_algorithms_comparison(statistics):
    for function in ["f", "q"]:
        for sigma in initial_sigmas:
            fig, ax = plt.subplots()
            x = mi_population_sizes
            for adaptation in ["LMR", "SA"]:
                fitnesses = []
                for mi in mi_population_sizes:
                    individual = get_stats_entry(statistics, function, mi, sigma, adaptation)
                    fitnesses.append(individual['best_fitness'])
                ax.plot(x, fitnesses, label=adaptation)
            ax.legend()
            ax.set_xlabel('Rozmiar populacji μ')
            ax.set_ylabel('Ocena najlepszego osobnika')
            ax.set_xticks(x)
            ax.set_yscale('log')
            plt.grid()
            fig.savefig("comparison/func={} sigma={}.png".format(function, sigma))
            plt.close(fig)


def draw_algorithms_convergence_comparison(statistics):
    for function in ["f", "q"]:
        fig, ax = plt.subplots(len(mi_population_sizes), len(initial_sigmas), sharex=True, sharey=True)
        for rowIx, sigma in enumerate(initial_sigmas):
            for colIx, mi in enumerate(mi_population_sizes):
                for adaptation in ["LMR", "SA"]:
                    individual = get_stats_entry(statistics, function, mi, sigma, adaptation)
                    function_values = individual['function_values']
                    x = np.arange(0, len(function_values), 1)
                    ax[rowIx][colIx].plot(x, function_values, label=adaptation)
                ax[rowIx][colIx].legend(fontsize=7, loc='lower left')
                ax[rowIx][colIx].set_title("mi={} sigma={}".format(mi, sigma), fontdict={'fontsize': 7})
                ax[rowIx][colIx].set_yscale('log')
                ax[rowIx][colIx].grid()
        fig.tight_layout()
        fig.savefig("comparison/conv func={}.png".format(function))
        plt.close(fig)


def generate_comparison(statistics):
    Path("./comparison").mkdir(exist_ok=True)
    generate_algorithms_comparison_tex(statistics)
    draw_algorithms_comparison(statistics)
    draw_algorithms_convergence_comparison(statistics)
    # draw_initial_steps_comparison(statistics)
    # draw_newtons_comparison(statistics)


def write_statistics_to_file(statistics):
    f = open('stats.json', "w")
    json.dump(statistics, f)
    f.close()


def read_statistics_from_file():
    f = open('stats.json', "r")
    statistics = json.load(f)
    f.close()
    return statistics


def main():
    # statistics = generate_statistics()
    # write_statistics_to_file(statistics)

    statistics = read_statistics_from_file()
    generate_comparison(statistics)


if __name__ == "__main__":
    main()
