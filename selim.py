import numpy as np

# States
STABLE = 0
GROWTH = 1
DECLINE = 2
states = [STABLE, GROWTH, DECLINE]
n_states = len(states)

# Actions
INVEST = 0
DO_NOT_INVEST = 1
actions = [INVEST, DO_NOT_INVEST]
n_actions = len(actions)

# Rewards
rewards = [-1, 1]

# State transition probabilities
state_transition_probs = np.array([
    [[0.5, 0.5], [0.7, 0.3], [0, 1]],
    [[0.5, 0.5], [0.7, 0.3], [0, 1]],
    [[0.3, 0.7], [0.5, 0.5], [0, 1]],
])

# Utility arrays
V = np.zeros((n_states, n_actions))
policy = np.zeros((n_states,), dtype=np.int32)

# Hyperparameters
discount_factor = 0.9
threshold = 1e-6

# Helper function to get the expected reward for a state and action
# Cette fonction calcule la récompense moyenne pour un état et une action donnée en prenant en compte les probabilités de transition vers les états suivants.
def get_expected_reward(state, action):
    expected_reward = 0
    for next_state in states:
        if next_state < state_transition_probs.shape[2]:
            prob = state_transition_probs[state, action, next_state]
            expected_reward += prob * rewards[action]
    return expected_reward

# Helper function to get the expected value for a state and action
# Cette fonction calcule la valeur attendue pour un état et une action donnée en prenant en compte les probabilités de transition vers les états suivants et la fonction de valeur attendue pour ces états suivants.
def get_expected_value(state, action):
    expected_value = 0
    for next_state in states:
        if next_state < state_transition_probs.shape[2]:
            prob = state_transition_probs[state, action, next_state]
            expected_value += prob * V[next_state, policy[next_state]]
    return expected_value
# Value iteration algorithm

def value_iteration():
    t=0
    while True:
        delta = 0
        for state in states:
            i=0
            action_values = [get_expected_reward(state, action) + discount_factor * get_expected_value(state, action) for action in actions]
            best_action = np.argmax(action_values)
            best_value = action_values[best_action]
            delta = max(delta, np.abs(V[state, policy[state]] - best_value))
            V[state, policy[state]] = best_value
            policy[state] = best_action
            print(best_value, end=" | ")
            print(best_action)
            i+=1
        print('')
        if delta < threshold:
            break

# Run the value iteration algorithm
value_iteration()

# Print the optimal policy
print("Optimal policy:", policy)