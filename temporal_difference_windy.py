import gym
from windy_gridworld import WindyGridworldEnv
import numpy as np
from collections import defaultdict
import itertools
import pprint
import matplotlib.pyplot as plt
import pandas as pd

env = WindyGridworldEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """Creates epsilon greedy policy
    Args: Q -> dictionary mapping states -> action probabilities (numpy array with length nA)
    epsilon -> epsilon greedy factor (prob of choosing a random action)
    nA -> Number of actions

    Returns policy function -> given an observation, return probability of choosing an action (numpy array of length nA)
    """

    def policy_fn(observation):
        best_action = np.argmax(Q[observation])
        actions = np.ones(nA, dtype=float)*epsilon / nA
        actions[best_action] += (1 - epsilon)
        return actions
    return policy_fn


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """Implements SARSA TD(0) Policy Evaluation
    Args: env: OpenAI Gym env
    num_episodes: number of episodes to watch
    discount_factor: how you value delayed rewards
    alpha: TD learning rate
    epsilon: epsilon-greedy factor (prob of choosing a random action)

    Returns: (Q, episode_length, episode_reward) where Q is a dictionary mapping states -> action probabilities
    episode_length -> time to finish an episode
    episode_reward -> reward at the end of an episode
    """

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Episode info
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    current_policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i in range(num_episodes):
        # Take the first action
        state = env.reset()
        probs = current_policy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)


        for t in itertools.count():
            # Take the current action
            next_step, reward, done, _ = env.step(action)

            # Pick the next action
            next_probs = current_policy(next_step)
            next_action = np.random.choice(np.arange(len(next_probs)), p=next_probs)

            # Update episode info
            episode_lengths[i] = t
            episode_rewards[i] += reward

            # TD update
            td_target = reward + discount_factor*Q[next_step][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha*td_delta

            if done:
                print(t)
                break

            action = next_action
            state = next_step

    return Q, episode_lengths, episode_rewards


def plot_episode_stats(episode_lengths, episode_rewards, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    return fig1, fig2


Q, episode_lengths, episode_rewards = sarsa(env, 200)

# pprint.pprint(Q)
plot_episode_stats(episode_lengths, episode_rewards)





