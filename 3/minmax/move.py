import math
import random


class Move:
    def __init__(self, x: int = -1, y: int = -1, score: float = -math.inf):
        self.x = x
        self.y = y
        self.score = score

    @staticmethod
    def random():
        values = random.sample([0, 1, 2], 2)
        return Move(values[0], values[1])
