import random
import gym

env_name = "CartPole-v1"
env = gym.make(env_name)
print("Observation space ->", env.observation_space)
print("Action space ->", env.action_space)

# ---------------------------------------------------------------------------------------------------------------------
class Agent:
    def __init__(self, c_env):
        self.action_size = c_env.action_space.n
        print("Action size:", self.action_size)

    def get_action(self, c_state):
                                    # See https://www.gymlibrary.ml/environments/classic_control/cart_pole/ for more details
        pole_angle = c_state[2]     # c_state[0] == cart position
                                    # c_state[1] == cart velocity
                                    # c_state[2] == pole angle
                                    # c_state[3] == pole angle velocity

        c_action = 0 if pole_angle < 0 else 1   # 0 == push cart to the left
                                                # 1 == push cart to the right

        return c_action
# ---------------------------------------------------------------------------------------------------------------------

agent = Agent(env)
state = env.reset()

for _ in range(200):

    action = agent.get_action(state)
    new_state, reward, done, info = env.step(action)

    env.render()


