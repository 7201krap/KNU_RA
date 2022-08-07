import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.mu  = Dense(action_dim, activation='tanh')
        self.std = Dense(action_dim, activation='softplus') # softplus function: https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
                                                            # softplus is a smooth approximation to the ReLU function.

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        mu  = self.mu(x)
        std = self.std(x)

        # 평균값을 [-action_bound, action_bound] 범위로 조정.
        mu = Lambda(lambda x: x * self.action_bound)(mu)

        return [mu, std]

class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.v  = Dense(1, activation='linear')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        v = self.v(x)

        return v

class A2Cagent(object):
    def __init__(self, env):
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.env = env
        self.state_dim = env.observation_space.shape[0] # 상태변수 차원
        self.action_dim = env.action_space.shape[0]     # 행동 차원
        self.action_bound = env.action_space.high[0]    # 행동의 최대 크기
        self.std_bound = [1e-2, 1.0]                    # 표준편차의 최솟값과 최댓값 설정

        print("state_dim:", self.state_dim)
        print("action_dim:", self.action_dim)
        print("action_bound:", self.action_bound)
        print("std_bound:", self.std_bound)

        self.actor = Actor(self.action_dim, self.action_bound)
        self.actor.build(input_shape=(None, self.state_dim))

        self.critic = Critic()
        self.critic.build(input_shape=(None, self.state_dim))

        self.actor.summary()
        self.critic.summary()

        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        self.save_epi_reward = []   # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action - mu)**2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def get_action(self, state):
        mu_a, std_a = self.actor(state)
        mu_a = mu_a.numpy()[0]
        std_a = std_a.numpy()[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)

        # print("action:", action)

        return action

    def actor_learn(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu_a, std_a = self.actor(states, training=True)
            log_policy_pdf = self.log_pdf(mu_a, std_a, actions)

            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_sum(-loss_policy)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    def critic_learn(self, states, td_targets):
        # td_target = reward + \gamma * v_{s_{t+1}}
        # loss = (td_target - v_{s_t})**2
        with tf.GradientTape() as tape:
            v_s_t = self.critic(states, training=True)
            loss = tf.reduce_mean(tf.square(td_targets - v_s_t))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

    def td_target(self, rewards, next_v_values, dones):
        y_i = np.zeros(next_v_values.shape)
        for i in range(next_v_values.shape[0]):
            if dones[i]:
                y_i[i] = rewards[i]
            else:
                y_i[i] = rewards[i] + self.GAMMA * next_v_values[i]
        return y_i

    def load_weights(self, path):
        self.actor.load_weights(path + 'pendulum_actor.h5')
        self.critic.load_weights(path + 'pendulum_critic.h5')

    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)

        return unpack

    def train(self, max_episode_num):

        for ep in range(int(max_episode_num)):
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [] ,[] ,[]
            step, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                action = self.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
                action = np.clip(action, -self.action_bound, self.action_bound)
                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                done = np.reshape(done, [1, 1])

                train_reward = (reward + 8) / 8

                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(train_reward)
                batch_next_state.append(next_state)
                batch_done.append(done)

                if len(batch_state) < self.BATCH_SIZE:
                    state = next_state[0]
                    episode_reward = episode_reward + reward[0]
                    step = step + 1
                    continue    # run the while loop again

                # the program reaches to this line when len(batch_state) == self.BATCH_SIZE
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                train_rewards = self.unpack_batch(batch_reward)
                next_states = self.unpack_batch(batch_next_state)
                dones = self.unpack_batch(batch_done)

                # clear the batch
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [] ,[] ,[]

                # compute TD-targets
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))    # 32 next_v_values
                td_targets = self.td_target(train_rewards, next_v_values.numpy(), dones)    # 32 td_targets

                self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                  tf.convert_to_tensor(td_targets, dtype=tf.float32))

                v_values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
                advantages = train_rewards + self.GAMMA * next_v_values - v_values

                self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                 tf.convert_to_tensor(actions, dtype=tf.float32),
                                 tf.convert_to_tensor(advantages, dtype=tf.float32))

                state = next_state[0]
                episode_reward = episode_reward + reward[0]
                step = step + 1

            print('Episode: ', ep+1, '| Step: ', step, '| Reward: ', episode_reward)
            self.save_epi_reward.append(episode_reward)

            if ep % 10 == 0:
                self.actor.save_weights('./save_weights/pendulum_actor.h5')
                self.critic.save_weights('./save_weights/pendulum_critic.h5')

        # after finishing all episodes, save the rewards that we have got for each episode.
        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print("Reward for each episode: ", self.save_epi_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()








