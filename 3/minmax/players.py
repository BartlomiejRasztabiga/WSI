import time

from . import Player, Move, Board, Strategy


class AIPlayer(Player):
    def __init__(self, sign: str, max_depth: int = 8):
        super(AIPlayer, self).__init__(sign, "AI")
        self.max_depth = max_depth + 1
        self.strategy = None
        assert 1 <= self.max_depth <= 9

    def set_strategy(self, strategy: Strategy):
        self.strategy = strategy

    def move(self, board: Board) -> None:
        if self.strategy is None:
            raise "No strategy set!"

        moves_left = board.get_valid_moves_count()
        if moves_left == board.get_max_moves_count():
            move = Move.random()
        else:
            move = self.strategy.next_move(board, self.max_depth, self)

        time.sleep(1)
        board.set_move(move, self.sign)


class HumanPlayer(Player):
    def __init__(self, sign: str):
        super(HumanPlayer, self).__init__(sign, "Human")

    def move(self, board: Board) -> None:
        is_move_valid = False
        while not is_move_valid:
            move_input = input('move (y, x): ').split()
            move = Move(int(move_input[0]), int(move_input[1]))

            is_move_valid = board.is_move_valid(move)

            if not is_move_valid:
                print('invalid move, try again')
                continue

            board.set_move(move, self.sign)
