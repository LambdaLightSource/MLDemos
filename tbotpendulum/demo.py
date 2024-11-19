import numpy as np
import pygame
import pygame.gfxdraw
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
tf.compat.v1.disable_eager_execution()
class CustomPendulumEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 700))
        pygame.display.set_caption("Pendulum")
        self.clock = pygame.time.Clock()
        self.framerate = 30
        self.origin = np.array([500, 180])
        self.arrow = np.array([[1, 0], [1, 150], [0, 152], [-1, 150], [-1, 0], [1, 0]]).astype(float)
        self.arrow = self.arrow / self.arrow.max()
        self.scalefactor = 240
        self.g = 9.81
        self.l = 1
        self.alpha = 0
        self.omega = 0
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.dt = 1.0 / self.framerate
        self.C = 0.9
        self.done = False
        self.action_limit = 2  
    def step(self, action):
        torque = action[0]  
        self.alpha = (-np.sin(self.theta) * self.g / self.l) + torque
        self.omega += self.alpha * self.dt
        self.omega *= self.C
        self.theta += self.omega * self.dt
        normalized_theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        target_theta = 0  
        
        angle_diff = normalized_theta - target_theta
        
        angular_velocity = self.omega
        
        reward = - (angle_diff ** 2 + 0.1 * angular_velocity ** 2 + 0.001 * (action[0] ** 2))
        done = abs(self.theta - np.pi) < 0.01
        state = np.array([np.sin(self.theta), np.cos(self.theta), self.omega])
        return state, reward, done, {}
    def reset(self):
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.omega = 0
        return np.array([np.sin(self.theta), np.cos(self.theta), self.omega])
    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        self.screen.fill((90, 90, 90))  # Clear screen
        arrow1 = np.array([self.arrow[:, 0] * self.scalefactor / 2, (self.arrow[:, 1] * self.l * self.scalefactor)]).T
        arrow_rot = np.array(self.rotxy(self.theta + np.pi, arrow1))
        arrow_tup = tuple(map(tuple, tuple((arrow_rot + self.origin).astype(int))))
        pygame.gfxdraw.filled_polygon(self.screen, arrow_tup, (255, 255, 255, 155))
        pygame.gfxdraw.aapolygon(self.screen, arrow_tup, (255, 255, 255, 250))
        pygame.gfxdraw.aacircle(self.screen, self.origin[0], self.origin[1], 5, (255, 0, 0))
        pygame.gfxdraw.filled_circle(self.screen, self.origin[0], self.origin[1], 5, (255, 0, 0))
        pygame.gfxdraw.filled_circle(self.screen, arrow_tup[2][0], arrow_tup[2][1], 15, (0, 255, 255, 250))
        pygame.gfxdraw.aacircle(self.screen, arrow_tup[2][0], arrow_tup[2][1], 15, (0, 255, 255, 250))
        pygame.display.flip()
    def close(self):
        pygame.quit()
    def rotxy(self, theta, arr):
        c, s = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([[c, -s], [s, c]])
        return np.dot(arr, rot_matrix)
    def sample_action(self):
        return np.random.uniform(-self.action_limit, self.action_limit, size=(1,))

def build_actor(state_shape, action_dim, max_torque=3):
    inputs = Input(shape=(state_shape,))
    x = Dense(256, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    last_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    outputs = Dense(action_dim, activation='tanh', kernel_initializer=last_init)(x) * max_torque
    model = Model(inputs, outputs)
    return model
def build_critic(state_shape, action_dim):
    action_input = Input(shape=(action_dim,))
    observation_input = Input(shape=(state_shape,))
    concatenated = Concatenate()([action_input, observation_input])
    x = Dense(256, activation='relu')(concatenated)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model([action_input, observation_input], outputs)
    return model, action_input

env = CustomPendulumEnv()
actor = build_actor(3, 1, max_torque=2)
critic, action_input = build_critic(3, 1)

target_actor = build_actor(3, 1, max_torque=2)
target_critic, _ = build_critic(3, 1)
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())
def update_target(target_model, source_model, tau=0.005):
    target_weights = target_model.get_weights()
    source_weights = source_model.get_weights()
    updated_weights = [tau * source + (1 - tau) * target for source, target in zip(source_weights, target_weights)]
    target_model.set_weights(updated_weights)
memory = SequentialMemory(limit=200000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=1, theta=0.15, mu=0.0, sigma=0.2, sigma_min=0.05, n_steps_annealing=10000)
agent = DDPGAgent(nb_actions=1, actor=actor, critic=critic, critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=500, nb_steps_warmup_actor=500, random_process=random_process, gamma=0.9, target_model_update=0.005)  # Adjusted gamma
actor_learning_rate = 0.001
critic_learning_rate = 0.002
agent.compile([Adam(learning_rate=critic_learning_rate), Adam(learning_rate=actor_learning_rate)], metrics=['mae'])
initial_experiences = 10000
observation = env.reset()
for _ in range(initial_experiences):
    action = env.sample_action()
    next_observation, reward, done, info = env.step(action)
    memory.append(observation, action, reward, done, training=True)
    observation = next_observation if not done else env.reset()

agent.fit(env, nb_steps=1000000, visualize=True, verbose=1, log_interval=5000)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history['episode_reward']))

env.close()