from abc import abstractmethod

from . import Board


class Player:
    def __init__(self, sign: str, label: str):
        self.sign = sign
        self.label = label

    @abstractmethod
    def move(self, board: Board):
        pass
