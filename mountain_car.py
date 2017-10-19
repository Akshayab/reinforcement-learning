import gym
import gridworld
import pprint
import numpy as np

# env = gym.make('MountainCar-v0')
# observation = env.reset()
# print(env.action_space)
#
# for i in range(2000):
#     env.render()
#     print(observation)
#     observation, reward, done, info = env.step(1 + pow(-1, i))
#     print(observation, reward, done, info)

grid = gridworld.GridworldEnv()


def policy_evaluation(policy, env, delta=0.0001, discount_factor=1.0):
    """Evaluate a Policy given the full dynamics of the environment.
    Args: policy: [S, A] matrix, env = environment with transition probabilities where
    where env.P[s][a] = (prob, next_state, reward, done),
    delta = change of value function,
    discount factor = how much we weight future rewards

    Returns: value of this policy
    """
    V = np.zeros(env.nS) # initialize V(s) to be zero for all s
    while True:
        current_delta = 0
        for s in range(env.nS):
            v = 0
            for a, prob in enumerate(policy[s]):
                trans_prob, next_state, reward, done = env.P[s][a][0]
                v += prob*trans_prob*(reward + discount_factor*V[next_state])
            V[s] = v
            current_delta = max(current_delta, v - V[s])

        if current_delta < delta:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_evaluation, discount_factor=1.0):
    """"Policy Improvement Algorithm (Greedy)
    Args:
        env = OpenAI gym environment
        policy_eval_fn = An evaluation function which takes 4 arguments

        Returns
        (policy, value)
        where policy is optimal policy and value is optimal value
        policy = matrix of shape [S, A] -> transition probability to go from s -> a
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA])/ env.nA

    while True:
        # Evaluate current policy
        V = policy_eval_fn(policy=policy, env=env, delta=0.0001, discount_factor=discount_factor)
        print(V)

        policy_stable = True
        # For all states
        for s in range(env.nS):
            # policy chosen the best action
            policy_best_action = np.argmax(policy[s])

            # initialize all actions to zero
            actions = np.zeros(env.nA)
            for a in range(env.nA):
                prob, next_state, reward, done = env.P[s][a][0]
                actions[a] = prob*(reward + discount_factor*V[next_state])

            best_action = np.argmax(actions)

            # Greedy update:
            if policy_best_action != best_action:
                policy_stable = False

            # Make the policy choose the best action with a prob of 1 for state S
            policy[s] = np.eye(env.nA)[best_action]

        if policy_stable:
            return policy, V


policy, v = policy_improvement(grid)
print(policy)

print(np.reshape(np.argmax(policy, axis=1), grid.shape))
