# Q-learning with keyboard inputs (deterministic)

import sys
import termios
import tty

import gym
import readchar


class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT
}

# register FrozenLake with is_slippery False.
gym.envs.registration.register(id='FrozenLake-v3',
                               entry_point='gym.envs.toy_text:FrozenLakeEnv',
                               kwargs={'map_name': '4x4',
                                       'is_slippery': False}
                               )

env = gym.make('FrozenLake-v3')
env.reset()

print("Game started")
print("Press the key to move your agent")

env.render()  # show the initial board


while True:
    print("-----------------------------------------------------------------------------")

    key = inkey()  # choose an action from keyboard
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]

    state, reward, done, info = env.step(action)

    env.render()  # show the board after the execution

    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)
    # State: 지금 에이전트의 위치가 어디인지를 나타냄
    # Action: 에이전트(나)가 어떤 input를 주었는지를 나타냄

    if done:
        print("Finished with reward", reward)
        break
