from random import randint
from time import perf_counter

import argparse
from collections import deque
from itertools import chain, combinations
from operator import attrgetter
from typing import Collection, NamedTuple


class Element(NamedTuple):
    weight: float
    value: float

    @property
    def ratio(self) -> float:
        return self.value / self.weight


def solve_naive(max_weight: float, elements: Collection[Element]) -> tuple[Element, ...]:
    all_combinations = chain.from_iterable(
        combinations(elements, i) for i in range(1, len(elements) + 1)
    )

    return max(
        filter(lambda c: sum(elem.weight for elem in c) <= max_weight, all_combinations),
        key=lambda c: sum(elem.value for elem in c)
    )


def solve_by_ratio(max_weight: float, elements: Collection[Element]) -> tuple[Element, ...]:
    weight_to_go = max_weight
    packed_elements: deque[Element] = deque()

    for elem in sorted(elements, key=attrgetter("ratio"), reverse=True):
        if elem.weight <= weight_to_go:
            packed_elements.append(elem)
            weight_to_go -= elem.weight

            if weight_to_go == 0:
                break

    return tuple(packed_elements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--max_weight", type=int, default=100)
    parser.add_argument("-c", "--count", type=int, default=26,
                        help="how many elements to choose from")
    parser.add_argument("-lv", "--lowest_value", type=int, default=1)
    parser.add_argument("-hv", "--highest_value", type=int, default=10)
    parser.add_argument("-lw", "--lowest_weight", type=int, default=1)
    parser.add_argument("-hw", "--highest_weight", type=int, default=10)
    args = parser.parse_args()

    elements = [
        Element(randint(args.lowest_weight, args.highest_weight),
                randint(args.lowest_value, args.highest_value))
        for _ in range(args.count)
    ]

    print("Starting naive...")
    start = perf_counter()
    s = solve_naive(args.max_weight, elements)
    end = perf_counter() - start
    print(f"Naive took {end:.2f} s")
    print(s)

    sw = sum(i.weight for i in s)
    sv = sum(i.value for i in s)
    print(f"Naive Solution has {len(s)} elements, weight {sw}, value {sv}")

    print("Starting by ratio...")
    start = perf_counter()
    s = solve_by_ratio(args.max_weight, elements)
    end = perf_counter() - start
    print(f"By ratio took {end:.2f} s")
    print(s)

    sw = sum(i.weight for i in s)
    sv = sum(i.value for i in s)
    print(f"By ratio Solution has {len(s)} elements, weight {sw}, value {sv}")


if __name__ == "__main__":
    main()
