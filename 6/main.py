import gym
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng

ENV_NAME = 'Taxi-v3'


def draw_chart(x_axis, y_axis, title, file_name, x_label, y_label):
    plt.plot(x_axis, y_axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_axis)
    plt.title(title)
    plt.savefig(file_name)
    plt.close()


def train(episodes, learning_rate, discount_rate, exploration_rate):
    rng = default_rng(seed=1)
    env = gym.make(ENV_NAME)

    q_table = np.zeros((500, 6))

    for episode in range(episodes):
        state = env.reset()

        while True:
            rnd = rng.uniform()
            if rnd > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)
            q_table[state, action] = q_table[state, action] + learning_rate * (
                    reward + discount_rate * np.max(q_table[new_state, :]) - q_table[state, action])

            state = new_state

            if done:
                break

    return q_table


def evaluate(q_table, episodes):
    env = gym.make(ENV_NAME)
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = np.argmax(q_table[state, :])
            new_state, reward, done, _ = env.step(action)
            state = new_state
            episode_reward += reward

            if done:
                rewards.append(episode_reward)
                break

    return np.mean(rewards)


def main():
    eval_episodes = 100
    optimal_episodes = 4000
    optimal_lr = 0.2
    optimal_dr = 1.0
    optimal_er = 0.5

    ER = [0.5, 0.1, 0.01, 0.001, 0.0001]
    EPISODES = [1000, 2000, 5000, 10000]
    LR = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01]
    DR = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01]

    print("comparing ER")
    x = []
    y = []
    for exploration_rate in ER:
        q_table = train(episodes=optimal_episodes,
                        learning_rate=optimal_lr,
                        discount_rate=optimal_dr,
                        exploration_rate=exploration_rate)
        avg_reward = evaluate(q_table, eval_episodes)
        x.append(exploration_rate)
        y.append(avg_reward)
        print(exploration_rate, avg_reward)
    draw_chart(x, y, 'Porównanie ER', 'er_comparison.png', 'exploration rate', 'średnia nagroda')

    print("comparing episodes")
    x = []
    y = []
    for episode in EPISODES:
        q_table = train(episodes=episode,
                        learning_rate=optimal_lr,
                        discount_rate=optimal_dr,
                        exploration_rate=optimal_er)
        avg_reward = evaluate(q_table, eval_episodes)
        x.append(episode)
        y.append(avg_reward)
        print(episode, avg_reward)
    draw_chart(x, y, 'Porównanie liczby epizodów', 'episodes_comparison.png', 'episodes count', 'średnia nagroda')

    print("comparing LR")
    x = []
    y = []
    for lr in LR:
        q_table = train(episodes=optimal_episodes,
                        learning_rate=lr,
                        discount_rate=optimal_dr,
                        exploration_rate=optimal_er)
        avg_reward = evaluate(q_table, eval_episodes)
        x.append(lr)
        y.append(avg_reward)
        print(lr, avg_reward)
    draw_chart(x, y, 'Porównanie LR', 'lr_comparison.png', 'learning rate', 'średnia nagroda')

    print("comparing DR")
    x = []
    y = []
    for dr in DR:
        q_table = train(episodes=optimal_episodes,
                        learning_rate=optimal_lr,
                        discount_rate=dr,
                        exploration_rate=optimal_er)
        avg_reward = evaluate(q_table, eval_episodes)
        x.append(dr)
        y.append(avg_reward)
        print(dr, avg_reward)
    draw_chart(x, y, 'Porównanie DR', 'dr_comparison.png', 'discount rate', 'średnia nagroda')


if __name__ == "__main__":
    main()
