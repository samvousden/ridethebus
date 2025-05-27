import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

import ride_bus
from ride_bus import RideTheBus

import random

import tensorflow as tf


class RideBusEnv(gym.Env):
    def __init__(self, verbose = False):
        super(RideBusEnv, self).__init__()
        self.verbose = verbose
        self.game = RideTheBus()
        self.action_space = spaces.Discrete(32)  # 16 cards × 2 (higher/lower)
        self.observation_space = spaces.Box(low=1, high=13, shape=(16,), dtype=np.int32)

    def reset(self):
        self.game = RideTheBus()
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.game.cardvals[c[0]] for c in self.game.board], dtype=np.int32)

    def step(self, action,):
        card_index = action // 2
        guess_val = action % 2
        board_card = self.game.board[card_index][0]
        highlow = "higher" if guess_val == 1 else "lower"

        card_before = board_card
        deck_before = self.game.deck.copy()
        
        correct = self.game.input_guess(board_card, highlow)
        reward = 1 if correct else -1
        done = len(self.game.deck) == 0

        if self.verbose:
            new_card = self.game.board[card_index][0]
            actual_drawn = list(set(deck_before) - set(self.game.deck))
            revealed_card = actual_drawn[0] if actual_drawn else new_card  # fallback
            print(f"\n🃏 Selected card index {card_index}: {card_before}")
            print(f"🤔 Guessed: {highlow}")
            print(f"🎲 New card drawn: {revealed_card}")
            print(f"🎯 Guess was {'correct ✅' if correct else 'wrong ❌'} (reward: {reward})")
            print(f"🏆 Current Score: {self.game.score}")
            self.render()

        return self._get_obs(), reward, done, {}

    def render(self):
        print("Board:")
        for i in range(0, 16, 4):
            row = self.game.board[i:i+4]
            print("  ".join(c[0] for c in row))
        print(f"Score: {self.game.score}\n")

env = RideBusEnv()
num_actions = env.action_space.n
state_shape = env.observation_space.shape
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
learning_rate = 1e-3
batch_size = 64
max_episodes = 1000
max_steps = 100
target_score = 0.8

# Q-network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=state_shape),
    tf.keras.layers.Lambda(lambda x: x / 13.0),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_actions)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

# Replay buffer
buffer = []
buffer_capacity = 10000

def sample_from_buffer():
    minibatch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
    return states, actions, rewards.astype(np.float32), next_states, dones.astype(np.float32)

# Training loop
all_rewards = []

for episode in range(max_episodes):
    render_this = (episode + 1) % 50 == 0  # Every 50 episodes
    env.verbose = render_this
    
    state = env.reset()
    total_reward = 0
    

    for step in range(max_steps):
        state_input = np.expand_dims(state, axis=0)
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            q_values = model.predict(state_input, verbose=0)
            action = np.argmax(q_values[0])

        next_state, reward, done, _ = env.step(action)
        buffer.append((state, action, reward, next_state, done))
        if len(buffer) > buffer_capacity:
            buffer.pop(0)

        state = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = sample_from_buffer()
            next_qs = model.predict(next_states, verbose=0)
            max_next_qs = np.max(next_qs, axis=1)
            targets = rewards + (1 - dones) * gamma * max_next_qs

            masks = tf.one_hot(actions, num_actions)
            with tf.GradientTape() as tape:
                q_values = model(states)
                q_action = tf.reduce_sum(q_values * masks, axis=1)
                loss = loss_fn(targets, q_action)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if done:
            break

    all_rewards.append(total_reward)
    avg_reward = np.mean(all_rewards[-100:])
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    print(f"Episode {episode+1}: Reward = {total_reward}, Avg(100) = {avg_reward:.2f}, Epsilon = {epsilon:.2f}")
    #if avg_reward >= target_score:
    #    print("✅ Performance threshold reached. Stopping training.")
    #    break

model.save("ride_bus_model.h5")

plt.figure(figsize=(10, 5))
plt.plot(all_rewards, label='Reward per Episode', color='blue')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title("DQN Agent Training Performance on Ride the Bus")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Moving average reward
window = 50
if len(all_rewards) >= window:
    moving_avg = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, label=f'{window}-Episode Moving Average', color='green')
    plt.title("Smoothed Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()