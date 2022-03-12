# Dummy Q-learning. Lab of lecture 3

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random


def rargmax(vector):
    """
    rargmax is argmax that returns a random arg-value among eligible maximum values.
    For example, if all values are 0, this function returns a random action.
    However, if the maximum value exists (ex. 0, 0, 0, 1), it returns a last action because
    every value is 0 except the last value 1.
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)


# register FrozenLake with is_slippery False.
register(id='FrozenLake-v3',
         entry_point='gym.envs.toy_text:FrozenLakeEnv',
         kwargs={'map_name': '4x4',
                 'is_slippery': False}
        )

env = gym.make('FrozenLake-v3')
print("Initial Environment")
env.render()    # visualise the table

Q = np.zeros([env.observation_space.n, env.action_space.n])     # initialise table with all zeros
                                                                # env.observation_space.n == 16
                                                                # env.action_space.n == 4
num_episodes = 2000     # set learning parameters

rList = []
for i in range(num_episodes):

    state = env.reset()     # reset environment and get first new observation

    rAll = 0
    done = False    # initial value is False because the game has not started yet.

    # the Q-table learning algorithm
    while not done:     # get out of while loop when it is done
        action = rargmax(Q[state, :])

        new_state, reward, done, _ = env.step(action)

        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll = rAll + reward
        state = new_state

    rList.append(rAll)

print('---------------------------------------------------------')
print("Success rate:", str(sum(rList)/num_episodes))
print()
print("Final Q-table values")
print("Left Down Right Up")
print(Q)

plt.bar(range(len(rList)), rList, color='blue')
plt.title("Dummy/Vanilla Q Learning")
plt.xlabel("Episodes")
plt.ylabel("Successful or not")
plt.show()



