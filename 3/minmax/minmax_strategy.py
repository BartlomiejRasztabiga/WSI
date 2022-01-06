import math

from . import Player, Board, Move, Strategy


class MinMaxStrategy(Strategy):
    def __init__(self, controlled_player: Player, other_player: Player):
        super().__init__(controlled_player, other_player)
        self.heuristic_board = [
            [3, 2, 3],
            [2, 4, 2],
            [3, 2, 3]
        ]

    def evaluate(self, state: Board):
        moves_left = state.get_valid_moves_count()

        if state.is_winning(self.controlled_player.sign):
            score = 100 + moves_left
        elif state.is_winning(self.other_player.sign):
            score = -100 - moves_left
        else:
            score = 0

        return score

    def heuristic(self, state: Board, current_player: Player):
        if current_player == self.controlled_player:
            return self.heuristic_value(state, current_player.sign)
        else:
            return -self.heuristic_value(state, current_player.sign)

    def heuristic_value(self, state: Board, player_sign: str):
        value = 0
        for x, row in enumerate(self.heuristic_board):
            for y, field_value in enumerate(row):
                if state.board[x][y] == player_sign:
                    value += field_value
        return value

    def next_move(self, state: Board, depth: int, player: Player) -> Move:
        if player == self.controlled_player:
            best = Move(score=-math.inf)
        else:
            best = Move(score=math.inf)

        if state.is_game_over():
            score = self.evaluate(state)
            return Move(score=score)

        if depth == 0:
            heuristic_score = self.heuristic(state, player)
            return Move(score=heuristic_score)

        for move in state.get_valid_moves():
            state.set_move(move, player.sign)
            move.score = self.next_move(state, depth - 1, self.get_next_player(player)).score
            state.reset_move(move)

            if player == self.controlled_player:
                if move.score > best.score:
                    best = move  # max
            else:
                if move.score < best.score:
                    best = move  # min

        return best

    def get_next_player(self, current_player: Player):
        if current_player == self.controlled_player:
            return self.other_player
        return self.controlled_player
