import minmax


def ai_vs_ai():
    ai_player = minmax.AIPlayer('O', 8)
    ai_player2 = minmax.AIPlayer('X', 8)

    ai_player.set_strategy(minmax.MinMaxStrategy(ai_player, ai_player2))
    ai_player2.set_strategy(minmax.MinMaxStrategy(ai_player2, ai_player))

    game = minmax.Game(ai_player, ai_player2)

    game.play()


def human_vs_ai():
    human_player = minmax.HumanPlayer('X')
    ai_player = minmax.AIPlayer('O', 8)
    ai_player.set_strategy(minmax.MinMaxStrategy(ai_player, human_player))

    game = minmax.Game(human_player, ai_player)

    second_player_starts = True
    game.play(second_player_starts)


def human_vs_human():
    human_player = minmax.HumanPlayer('O')
    human_player1 = minmax.HumanPlayer('X')

    game = minmax.Game(human_player, human_player1)

    game.play()


def main():
    ai_vs_ai()
    # human_vs_ai()
    # human_vs_human()


if __name__ == "__main__":
    main()
