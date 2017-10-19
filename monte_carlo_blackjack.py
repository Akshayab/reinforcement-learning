from blackjack import BlackjackEnv
from collections import defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pprint

env = BlackjackEnv()
NUM_EPISODES = 10


def strategy(observation):
    score, dealer_score, usable_ace = observation
    if score >= 20:
        return 0  # stay

    return 1  # hit


def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {}, Usable Ace: {}, Dealer Score: {}".format(score, usable_ace, dealer_score))


##### POLICY ESTIMATION ########


def mc_prediction(env, policy, num_episodes, discount_factor=1.0):
    """Returns an MC estimation for the value function of a given policy
    Args: env = openAI gym env
    policy = a function that maps observation to action probabilities
    num_episodes = # of times to sample
    discount_factor = delayed reward factor

    Returns a dictionary that maps state -> Value (array of size env.nS)
    """

    # sum and number of times you have seen it
    return_sum = defaultdict(float)
    return_count = defaultdict(float)

    V = defaultdict(float)  # final value function = avg of above things

    for i in range(num_episodes):
        # Generate episodes
        episode = []
        state = env.reset()

        for t in range(100):
            action = strategy(state)
            next_step, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_step

        states_in_episode = set([data[0] for data in episode])
        for state in states_in_episode:
            # Get first occurrence
            first_occurrence_idx = [i for i, x in enumerate(episode) if x[0] == state][0]

            # sum up everything since first occurrence
            G = sum([x[2]*discount_factor**i for i,x in enumerate(episode[first_occurrence_idx:])])

            return_sum[state] += G
            return_count[state] += 1
            V[state] = return_sum[state] / return_count[state]

    return V

#
# for episode in range(NUM_EPISODES):
#     observation = env.reset()
#     for t in range(100):
#         print_observation(observation)
#         action = strategy(observation)
#         print("Taking Action {}".format(['Stick', 'Hit'][action]))
#         observation, reward, done, _ = env.step(action)
#         if done:
#             print("Reward {}".format(reward))
#             print(observation)
#             break


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Separate no ace and ace cases
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


# V_100k = mc_prediction(env, strategy, num_episodes=100000)
# pprint.pprint(V_100k)
# plot_value_function(V_100k, "100k")


##### MC Control ########

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """Creates an epsilon-greedy policy
    Args: Q: dictionary mapping state -> action values (each value is a numpy array of length (nA))
    epsilon: prob of selecting a random action A
    nA: # of actions in this env.
    Returns a function that takes observation as an input and returns probabilities of choosing actions
    """

    def policy_fn(observation):
        A = np.ones(nA)*epsilon/nA
        best = np.argmax(Q[observation])
        A[best] += (1 - epsilon)
        return A

    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """Find an optimal policy using MC epsilon-greedy method
    Args: env - openAi env
    num_episodes - episodes to alternate between improvement and evaluation
    discount_factor - weight of delayed rewards
    epsilon - e-greeedy factor (prob of selecting random action A)

    returns (Q, policy) where Q is the dictionary mapping states -> action values (numpy array of length nA)
    and policy is function which takes input observation and outputs probablity of taking actions (numpy array of length nA)
    """

    # Sums and averages
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # Action-Value function
    # maps state -> numpy array of actions
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Current policy
    current_policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # for num_episodes - estimate value of Q
    for i in range(num_episodes):
        # This holds (state, next_action, reward) that we see
        episode = []
        state = env.reset()
        for i in range(100):
            probs = current_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all s,a pairs
        sa_in_episode = set([(x[0], x[1]) for x in episode])

        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_idx = [i for i, x in enumerate(episode) if x[0] == state and x[1] == action][0]
            G = sum(x[2]*discount_factor**i for i, x in enumerate(episode[first_idx:]))

            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1
            Q[state][action] = returns_sum[sa_pair]/returns_count[sa_pair]

    return Q, current_policy


# Q, policy = mc_control_epsilon_greedy(env, num_episodes=100000, epsilon=0.1)
#
# # Plot by choosing best action at every state
# V = defaultdict(float)
# for state, actions in Q.items():
#     action = actions.max()
#     V[state] = action
#
# pprint.pprint(V)
# plot_value_function(V, title="Optimal Value Function")


##### Off-line policy ########

def create_random_policy(nA):
    """ Creates a random policy fn
    Args: nA -> number of actions
    Returns -> a policy function that takes an observation as argument and returns probabilities of actions as np array
    """
    actions = np.ones(nA)/ nA
    def policy_fn(observation):
        return actions
    return policy_fn


def create_greedy_policy(Q):
    """Given a action-value fn, generate a policy fn
    Args: Q -> A dictionary mapping states -> actions values

    Returns -> a policy function that takes an observation as argument and returns probabilities of actions as np array
    """

    def policy_fn(observation):
        best_action = np.argmax(Q[observation])
        actions = np.zeros(len(Q[observation]))
        actions[best_action] = 1.0
        return actions

    return policy_fn


def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """Returns an optimal policy using MC off-line control
    Args: Q -> env: OpenAI gym environment
    num_episodes: # of episodes to estimate value
    behavior_policy: behavior to follow while generating episodes
    discount_factor: delayed reward factor

    Returns (Q, policy) where Q is a dictionary mapping states -> actions value
    policy -> a function that takes observation as an argument and returns prob of actions to take as np array
    """

    # Final action-value dict
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # cumalative denominator
    D = defaultdict(lambda: np.zeros(env.action_space.n))

    # Target policy
    target_policy = create_greedy_policy(Q)


    for i in range(num_episodes):
        # All (state, action, reward that we see)
        episode = []
        state = env.reset()

        for t in range(100):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_step, reward, done, _ = env.step(action)

            episode.append((state, action, reward))
            if done:
                break
            state = next_step


        # Sum of discounted rewards
        G = 0.0

        # Weights of return
        W = 1.0

        # For each step, backwards
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]

            # Update discount
            G = discount_factor* G + reward

            # Update weighted importance sampling formula denominator
            D[state][action] += W

            Q[state][action] += (W / D[state][action]) * (G - Q[state][action])

            if action != np.argmax(target_policy(state)):
                break

            W = W*1./behavior_policy(state)[action]

    return Q, target_policy


random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=100000, behavior_policy=random_policy)


# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plot_value_function(V, title="Optimal Value Function")
