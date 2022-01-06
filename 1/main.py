import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from minimizer import Minimizer, get_initial_point

SIMPLE_GRADIENT_DESCENT = "Simple Gradient Descent"
NEWTONS_DESCENT_WITH_CONSTANT_STEP = "Newtons Descent with Constant Step"
NEWTONS_DESCENT_WITH_ADAPTIVE_STEP = "Newtons Descent with Adaptive Step"

N = [10, 20]
A = [1, 10, 100]
STEP = [0.01, 0.1, 0.5, 1.0, 1.5, 1.6, 1.7, 1.8, 1.9]
ALGOS = [SIMPLE_GRADIENT_DESCENT, NEWTONS_DESCENT_WITH_CONSTANT_STEP, NEWTONS_DESCENT_WITH_ADAPTIVE_STEP]


def get_function_to_minimize(a, n):
    def f(x):
        assert len(x) == n
        y = 0
        for i in range(1, n + 1):
            y += a ** ((i - 1) / (n - 1)) * x[i - 1] ** 2
        return y

    return f


def generate_statistics():
    tolerance = 0.0001
    max_number_of_iterations = 5000

    print("Running minimizer to get some statistics, tolerance={}".format(tolerance))
    statistics = []
    for n in N:
        initial_point = get_initial_point(n)
        for a in A:
            for step in STEP:
                f = get_function_to_minimize(a, n)
                minimizer = Minimizer(f, step, tolerance, max_number_of_iterations, initial_point)
                algorithms = [
                    (minimizer.simple_gradient_descent, SIMPLE_GRADIENT_DESCENT),
                    (minimizer.newton_descent, NEWTONS_DESCENT_WITH_CONSTANT_STEP),
                    (minimizer.newton_descent_adaptive_step, NEWTONS_DESCENT_WITH_ADAPTIVE_STEP)]
                for algorithm, algorithm_name in algorithms:
                    print("{:<35} n={:<2} a={:<3} step={:<4}".format(algorithm_name, n, a, step), end=" - ")
                    stats = {"algorithm": algorithm_name, "a": a, "n": n, "step": step}
                    result = algorithm()
                    stats["converged"] = result[0][0]
                    stats["ending_point"] = result[0][1].tolist()
                    stats["function_values"] = result[0][2]
                    stats["iterations"] = result[0][3]
                    stats["time_elapsed"] = result[1]
                    if stats["converged"]:
                        print("{:<3} iterations {:0.2f} seconds elapsed".format(stats["iterations"],
                                                                                stats["time_elapsed"]))
                    else:
                        print("Didn't converge in {} iterations".format(stats["iterations"]))
                    statistics.append(stats)
                print("=" * 100)

    return statistics


def get_stats(statistics, n, a, step):
    return list(filter(lambda stat: stat['n'] == n and stat['a'] == a and stat['step'] == step, statistics))


def get_max_iteration(statistics):
    return max(statistics, key=lambda stat: stat['iterations'] if 'iterations' in stat else 0)['iterations']


def pad_function_values_with_nones(function_values, max_iteration):
    return np.pad(np.array(function_values), (0, max_iteration - len(function_values)), mode='constant',
                  constant_values=None)


def generate_x_axis(max_iteration):
    return np.arange(1, max_iteration + 1, 1)


def get_algorithm_statistics(statistics, algorithm_name):
    return list(filter(lambda stat: stat['algorithm'] == algorithm_name, statistics))[0]


def set_x_axis_to_use_int_values():
    gca = plt.gca()
    gca.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))


def draw_algorithms_comparison(statistics):
    for n in N:
        for a in A:
            for step in STEP:
                stats = get_stats(statistics, n, a, step)
                max_iteration = get_max_iteration(stats)
                x = generate_x_axis(max_iteration)
                fig, ax = plt.subplots()
                for algo_name in ALGOS:
                    algo_stats = get_algorithm_statistics(stats, algo_name)
                    function_values = algo_stats['function_values']
                    function_values_padded_with_nones = pad_function_values_with_nones(function_values, max_iteration)
                    set_x_axis_to_use_int_values()
                    ax.plot(x, function_values_padded_with_nones, label=algo_name)
                ax.legend()
                ax.set_xlabel('numer iteracji')
                ax.set_ylabel('wartość f(x)')
                ax.set_title("n={} a={} step={}".format(n, a, step))
                fig.savefig("algorithms_comparison/n={} a={} step={}.png".format(n, a, step))
                plt.close(fig)


def draw_initial_steps_comparison(statistics):
    n = 10
    a = 1

    for algo_name in ALGOS:
        fig, ax = plt.subplots()

        for step in STEP:
            stats = get_stats(statistics, n, a, step)
            max_iteration = get_max_iteration(stats)
            x = generate_x_axis(max_iteration)

            algo_stats = get_algorithm_statistics(stats, algo_name)
            function_values = algo_stats['function_values']
            function_values_padded_with_nones = pad_function_values_with_nones(function_values, max_iteration)
            set_x_axis_to_use_int_values()
            ax.plot(x, function_values_padded_with_nones, label=step)
        ax.legend()
        ax.set_xlabel('numer iteracji')
        ax.set_ylabel('wartość f(x)')
        ax.set_title("Wpływ początkowego kroku na liczbę iteracji\n({})".format(algo_name))

        fig.savefig("algorithms_comparison/steps_comparison_{}.png".format(algo_name))
        plt.close(fig)


def draw_newtons_comparison(statistics):
    n = 10
    a = 1

    for step in STEP:
        fig, ax = plt.subplots()

        for algo_name in [NEWTONS_DESCENT_WITH_CONSTANT_STEP, NEWTONS_DESCENT_WITH_ADAPTIVE_STEP]:
            stats = get_stats(statistics, n, a, step)
            max_iteration = get_max_iteration(stats)
            x = generate_x_axis(max_iteration)

            algo_stats = get_algorithm_statistics(stats, algo_name)
            function_values = algo_stats['function_values']
            function_values_padded_with_nones = pad_function_values_with_nones(function_values, max_iteration)
            set_x_axis_to_use_int_values()
            ax.plot(x, function_values_padded_with_nones, label=algo_name)
        ax.legend()
        ax.set_xlabel('numer iteracji')
        ax.set_ylabel('wartość f(x)')
        ax.set_title("Porównanie wariantów algorytmu Newtona (step={})".format(step))

        fig.savefig("algorithms_comparison/newtons_comparison_step={}.png".format(step))
        plt.close(fig)


def draw_graphs(statistics):
    Path("./algorithms_comparison").mkdir(exist_ok=True)
    draw_algorithms_comparison(statistics)
    draw_initial_steps_comparison(statistics)
    draw_newtons_comparison(statistics)


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
    statistics = generate_statistics()
    # write_statistics_to_file(statistics)

    # statistics = read_statistics_from_file()
    draw_graphs(statistics)


if __name__ == "__main__":
    main()
