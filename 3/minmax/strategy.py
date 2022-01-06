from abc import abstractmethod

from . import Player, Board, Move


class Strategy:
    def __init__(self, controlled_player: Player, other_player: Player):
        self.controlled_player = controlled_player
        self.other_player = other_player

    @abstractmethod
    def next_move(self, state: Board, depth: int, player: Player) -> Move:
        pass
