import gym
from clif_walking import CliffWalkingEnv
import numpy as np
from collections import defaultdict
import itertools
import pprint
import matplotlib.pyplot as plt
import pandas as pd

env = CliffWalkingEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """Returns an epsilon greedy policy
    Args: Q -> dictionary mapping states -> action probabilities
    epsilon -> epsilon greedy factor (probability of randomly choosing an action)
    nA -> number of Actions
    :return: A Function which takes an observation and returns a probability of taking action (np array size nA)
    """

    def policy_fn(observation):
        actions = np.ones(nA, dtype=float)*epsilon / nA
        best_action = np.argmax(Q[observation])
        actions[best_action] += (1 - epsilon)
        return actions
    return policy_fn


def q_learning(env, num_episodes, epsilon=0.1, discount_factor=1.0, alpha=0.5):
    """Implements Q-learning (or offline TD learning)
    Args: env -> OpenAI gym environment
    num_episodes: num of episodes to run this for
    epsilon: probablity of chosing a random action
    discount_factor: factor to weigh delayed rewards
    alpha: learning rate

    Returns (Q, episode_lengths)
    Q is a mapping of state to probability of taking actions
    episode_lengths is an np array for time it takes for an episode to finish
    episode_rewards is an np array of rewards per episode
    """

    # Q function
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Episode lengths
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # current policy (that we are following)
    current_policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i in range(num_episodes):
        state = env.reset()

        for t in itertools.count():
            action_probs = current_policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_step, reward, done, _ = env.step(action)

            best_next_action = np.max(Q[next_step])
            Q[state][action] += alpha*(reward + discount_factor*best_next_action - Q[state][action])

            episode_rewards[i] += reward

            if done:
                episode_lengths[i] = t
                break
            state = next_step

    return Q, episode_lengths, episode_rewards


def plot(episode_lengths, episode_rewards, smoothing_window=10):
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.show()

    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")   
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show()

    return fig1, fig2


Q, episode_lengths, episode_rewards = q_learning(env, 1000)
pprint.pprint(Q)
plot(episode_lengths, episode_rewards)
