import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from ride_bus_env import RideBusEnv





env = RideBusEnv(verbose=False)
num_actions = env.action_space.n
state_shape = env.observation_space.shape
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.0
learning_rate = 1e-3
batch_size = 64
max_episodes = 1000
min_episodes_before_stop = 100
max_steps = 20
target_score = 12
target_update_freq = 10

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=state_shape),
        tf.keras.layers.Lambda(lambda x: x / 13.0),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_actions)
    ])

model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

buffer = []
buffer_capacity = 10000

@tf.function
def train_step(states, actions, targets):
    with tf.GradientTape() as tape:
        q_values = model(states)
        q_action = tf.reduce_sum(q_values * tf.one_hot(actions, num_actions), axis=1)
        loss = loss_fn(targets, q_action)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def sample_from_buffer():
    minibatch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
    return (
        states.astype(np.float32) / 13.0,
        actions,
        rewards.astype(np.float32),
        next_states.astype(np.float32) / 13.0,
        dones.astype(np.float32)
    )

all_rewards = []

for episode in range(max_episodes):
    state = env.reset()
    total_reward = 0
    guess_counts = {"higher": 0, "lower": 0}

    for step in range(max_steps):
        state_input = np.expand_dims(state, axis=0).astype(np.float32) / 13.0

        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            q_values = model.predict(state_input, verbose=0)
            action = np.argmax(q_values[0])

        card_index = action // 2
        guess_val = action % 2
        guess_counts["higher" if guess_val else "lower"] += 1

        next_state, (val_card, val_new), done, info = env.step(action)

        margin = val_new - val_card
        if info["guess_val"] == 0:
            margin = -margin
        adjusted_reward = np.sign(margin)

        buffer.append((state, action, adjusted_reward, next_state, done))
        if len(buffer) > buffer_capacity:
            buffer.pop(0)

        state = next_state
        total_reward += adjusted_reward

        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = sample_from_buffer()
            max_next_qs = np.max(target_model.predict(next_states, verbose=0), axis=1)
            targets = rewards + (1 - dones) * gamma * max_next_qs
            train_step(states, actions, targets)

        if done:
            break

    all_rewards.append(total_reward)
    avg_reward = np.mean(all_rewards[-100:])
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if (episode + 1) % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    if (episode + 1) % 10 == 0 or episode == 0:
        print(f"Ep {episode+1}: Reward = {total_reward}, Avg(100) = {avg_reward:.2f}, Eps = {epsilon:.2f}")
        print(f"Guess split: {guess_counts}")

    if episode + 1 >= min_episodes_before_stop and avg_reward >= target_score:
        print("✅ Stopping: Agent reached target performance.")
        break

model.save("ridebus_dqn_model.keras")

# Plot rewards
plt.figure(figsize=(10, 5))
plt.plot(all_rewards, label='Reward per Episode', color='blue')
plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward per Episode")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

if len(all_rewards) >= 50:
    moving_avg = np.convolve(all_rewards, np.ones(50)/50, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, label='50-Episode Moving Average', color='green')
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Smoothed Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Evaluation (Summary Only) ---
print("\n🔍 Final Evaluation Summary (10 episodes):")
eval_rewards = []
for i in range(10):
    eval_env = RideBusEnv(verbose=False)
    state = eval_env.reset()
    done = False
    episode_reward = 0
    while not done:
        state_input = np.expand_dims(state, axis=0).astype(np.float32) / 13.0
        q_values = model.predict(state_input, verbose=0)
        action = np.argmax(q_values[0])
        state, (val_card, val_new), done, info = eval_env.step(action)

        margin = val_new - val_card
        if info["guess_val"] == 0:
            margin = -margin
        adjusted_reward = np.sign(margin)
        episode_reward += adjusted_reward

    eval_rewards.append(episode_reward)
    print(f"Episode {i+1}: Total Reward = {episode_reward}")

avg_eval = np.mean(eval_rewards)
print(f"\n📊 Average Total Reward Over 10 Evaluation Episodes: {avg_eval:.2f}")