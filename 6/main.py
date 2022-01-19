import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


def run(episodes=2000, learning_rate=0.03, discount_rate=0.97, exploration_rate=1.0):
    rng = default_rng(seed=1)
    env = gym.make('Taxi-v3')

    actions_space_size = env.action_space.n
    observations_space_size = env.observation_space.n

    q_table = np.zeros((observations_space_size, actions_space_size))

    rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

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
            episode_reward += reward

            if done:
                rewards.append(episode_reward)
                break

    plt.plot(range(episodes), rewards)
    plt.title('Rewards')
    plt.grid()
    plt.savefig(f"episodes={episodes} lr={learning_rate} dr={discount_rate} er={exploration_rate}.png")
    plt.close()

    return np.mean(rewards[-100:])


def main():
    optimal_episodes = 5000
    optimal_lr = 0.03
    optimal_dr = 0.97
    optimal_er = 0.001

    print("comparing ER")
    for exploration_rate in [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]:
        avg_reward = run(episodes=optimal_episodes,
                         learning_rate=optimal_lr,
                         discount_rate=optimal_dr,
                         exploration_rate=exploration_rate)
        print(exploration_rate, avg_reward)
    print()

    print("comparing episodes")
    for episode in [100, 1000, 2000, 5000, 10000]:
        avg_reward = run(episodes=episode,
                         learning_rate=optimal_lr,
                         discount_rate=optimal_dr,
                         exploration_rate=optimal_er)
        print(episode, avg_reward)
    print()

    print("comparing LR")
    for lr in [2.0, 1.0, 0.5, 0.1, 0.01, 0.001]:
        avg_reward = run(episodes=optimal_episodes,
                         learning_rate=lr,
                         discount_rate=optimal_dr,
                         exploration_rate=optimal_er)
        print(lr, avg_reward)
    print()

    print("comparing DR")
    for dr in [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]:
        avg_reward = run(episodes=optimal_episodes,
                         learning_rate=optimal_lr,
                         discount_rate=dr,
                         exploration_rate=optimal_er)
        print(dr, avg_reward)
    print()


if __name__ == "__main__":
    main()
