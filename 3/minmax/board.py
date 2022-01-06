from typing import List

from . import Move

EMPTY_CELL = None


class Board:
    def __init__(self, player1_sign: str, player2_sign: str, rows_count: int = 3, columns_count: int = 3):
        self.board = [[EMPTY_CELL for _ in range(columns_count)] for _ in range(rows_count)]
        self.player1_sign = player1_sign
        self.player2_sign = player2_sign
        self.rows_count = rows_count
        self.columns_count = columns_count

    def is_move_valid(self, move: Move) -> bool:
        for valid_move in self.get_valid_moves():
            if valid_move.x == move.x and valid_move.y == move.y:
                return True
        return False

    def get_max_moves_count(self) -> int:
        return self.rows_count * self.columns_count

    def get_valid_moves_count(self) -> int:
        return len(self.get_valid_moves())

    def get_valid_moves(self) -> List[Move]:
        moves = []
        for x, row in enumerate(self.board):
            for y, cell in enumerate(row):
                if cell == EMPTY_CELL:
                    moves.append(Move(x, y))
        return moves

    def set_move(self, move: Move, sign: str) -> None:
        self.board[move.x][move.y] = sign

    def reset_move(self, move: Move) -> None:
        self.board[move.x][move.y] = EMPTY_CELL

    def is_game_over(self) -> bool:
        return self.get_valid_moves_count() == 0 or self.is_winning(self.player1_sign) or self.is_winning(
            self.player2_sign)

    def is_winning(self, player_sign: str) -> bool:
        winning_states = self.get_winning_states()
        for state in winning_states:
            if all(sign == player_sign for sign in state):
                return True
        return False

    def get_winning_states(self) -> List[List[str]]:
        winning_states = []

        # rows
        for x in range(self.rows_count):
            winning_state = []
            for y in range(self.columns_count):
                winning_state.append(self.board[x][y])
            winning_states.append(winning_state)

        # columns
        for y in range(self.columns_count):
            winning_state = []
            for x in range(self.rows_count):
                winning_state.append(self.board[x][y])
            winning_states.append(winning_state)

        # first diagonal
        winning_state = []
        for x, y in zip(range(self.rows_count), range(self.columns_count)):
            winning_state.append(self.board[x][y])
        winning_states.append(winning_state)

        # second diagonal
        winning_state = []
        for x, y in zip(range(self.rows_count - 1, -1, -1), range(self.columns_count)):
            winning_state.append(self.board[x][y])
        winning_states.append(winning_state)

        return winning_states

    def print(self) -> None:
        print('=' * 15)
        for row in self.board:
            for cell in row:
                print(f'| {cell if cell is not None else " "} |', end='')
            print()
