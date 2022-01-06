from . import Player, Board, clean_screen


class Game:
    def __init__(self, player1: Player, player2: Player):
        self.board = Board(player1.sign, player2.sign)
        self.player1 = player1
        self.player2 = player2

    def play(self, second_player_starts: bool = False):
        if second_player_starts:
            self.player_turn(self.player2)

        while not self.board.is_game_over():
            self.player_turn(self.player1)
            self.player_turn(self.player2)

        clean_screen()
        self.board.print()

        if self.board.is_winning(self.player1.sign):
            print(f"Player {self.player1.label} ({self.player1.sign}) has won!")
        elif self.board.is_winning(self.player2.sign):
            print(f"Player {self.player2.label} ({self.player2.sign}) has won!")
        else:
            print("DRAW!")

    def player_turn(self, player: Player):
        if self.board.is_game_over():
            return

        clean_screen()
        print(f'{player.label} turn [{player.sign}]')
        self.board.print()

        player.move(self.board)
