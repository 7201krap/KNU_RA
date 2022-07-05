# Q-learning with e-greedy algorithm. Lab of lecture 4

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random

# register FrozenLake with is_slippery False.
register(id='FrozenLake-v3',
         entry_point='gym.envs.toy_text:FrozenLakeEnv',
         kwargs={'map_name': '4x4',
                 'is_slippery': False}
        )

env = gym.make('FrozenLake-v3')
env.render()

Q = np.zeros([env.observation_space.n, env.action_space.n])     # initialise table with all zeros
                                                                # env.observation_space.n == 16
                                                                # env.action_space.n == 4
dis = .99
num_episodes = 2000

rList = []
for i in range(num_episodes):

    state = env.reset()     # reset environment and get first new observation

    rAll = 0
    done = False    # initial value is False because the game has not started yet.

    e = 1. / ((i // 100) + 1)

    # the Q-table learning algorithm
    while not done:     # get out of while loop when it is done

        # <Exploit and Exploration algorithm> Choose an action with E-greedy algorithm
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # <Discounted reward> Update Q-table with new knowledge using decay rate
        Q[state, action] = reward + dis * np.max(Q[new_state, :])

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
plt.title("Dummy/Vanilla Q Learning with discount factor")
plt.xlabel("Episodes")
plt.ylabel("Successful or not")
plt.show()
