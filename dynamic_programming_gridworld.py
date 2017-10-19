import gridworld
import numpy as np

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
                for trans_prob, next_state, reward, done in env.P[s][a]:
                    v += prob*trans_prob*(reward + discount_factor*V[next_state])
            current_delta = max(current_delta, np.abs(v - V[s]))
            V[s] = v

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

        policy_stable = True
        # For all states
        for s in range(env.nS):
            # policy chosen the best action
            policy_best_action = np.argmax(policy[s])

            # initialize all actions to zero
            actions = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    actions[a] += prob*(reward + discount_factor*V[next_state])

            best_action = np.argmax(actions)

            # Greedy update:
            if policy_best_action != best_action:
                policy_stable = False

            # Make the policy choose the best action with a prob of 1 for state S
            policy[s] = np.eye(env.nA)[best_action]

        if policy_stable:
            return policy, V


def value_iteration(env, delta=0.0001, discount_factor=1.0):
    """"Uses value iteration to determine an optimal policy
    Args: openAI gym env, delta -> decides when to stop, discount_Factor = how much preference to delayed rewards
    Returns: optimal policy, optimal value
    policy -> probability distribution with shape [S, A]
    """

    def one_step_lookahead(state, V):

        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob*(reward + discount_factor*V[next_state])
        return A

    V = np.zeros(env.nS)

    while True:
        current_delta = 0

        # For each state
        for s in range(env.nS):
            # Do a one step lookahead to get the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)

            # Delta across all states
            current_delta = max(current_delta, abs(V[s] - best_action_value))
            V[s] = best_action_value

        if current_delta < delta:
            break

    # Create a deterministic policy
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s][best_action] = 1

    return policy, V


policy, v = policy_improvement(grid)
policy_val, v_val = value_iteration(grid)

print(np.reshape(np.argmax(policy, axis=1), grid.shape))
print(np.reshape(np.argmax(policy_val, axis=1), grid.shape))
