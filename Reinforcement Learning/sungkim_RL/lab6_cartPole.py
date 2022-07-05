import gym
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

env = gym.make('CartPole-v0')
env.reset()

random_episodes = 0
reward_sum = 0

while random_episodes < 10:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

    print(f"observation:{observation}, reward:{reward}, done:{done}")

    reward_sum += reward

    if done:
        random_episodes += 1
        print(f"Episode {random_episodes}: Reward for this episode == {reward_sum}")
        reward_sum = 0
        env.reset()


