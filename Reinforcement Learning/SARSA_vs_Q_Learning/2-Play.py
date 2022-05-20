import random
import gym

env_name = "CartPole-v1"
env_name = "MountainCar-v0"

env = gym.make(env_name)
print("Observation space ->", env.observation_space)
print("Action space ->", env.action_space)

# ---------------------------------------------------------------------------------------------------------------------
class Agent:
    def __init__(self, c_env):
        self.action_size = c_env.action_space.n
        print("Action size:", self.action_size)

    def get_action(self, c_state):
        c_action = random.choice(range(self.action_size))
        return c_action
# ---------------------------------------------------------------------------------------------------------------------

agent = Agent(env)
state = env.reset()

for _ in range(200):

    action = agent.get_action(state)
    new_state, reward, done, info = env.step(action)

    env.render()

