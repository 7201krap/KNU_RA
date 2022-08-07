import gym
import tensorflow as tf
from a2c_learn import A2Cagent

def main():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    agent = A2Cagent(env)
    agent.load_weights('./save_weights/')

    step = 0
    state = env.reset()

    while True:
        env.render()

        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]
        next_state, reward, done, _ = env.step(action)
        state = next_state

        print('Step:', step, 'Reward:', reward)
        step = step + 1

        if done:
            break

    env.close()

if __name__=="__main__":
    main()