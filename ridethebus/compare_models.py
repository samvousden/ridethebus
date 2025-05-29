# Compare performance of standard DQN vs Dueling DQN on Ride the Bus

import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, stdev
from math import sqrt
from ride_bus_env import RideBusEnv

# Register the functions used in Lambda layers
@register_keras_serializable()
def normalize_input(x):
    return x / 13.0

@register_keras_serializable()
def compute_advantage_mean(a):
    return tf.reduce_mean(a, axis=1, keepdims=True)

tf.keras.config.enable_unsafe_deserialization()

# Load both models
standard_model = tf.keras.models.load_model("ridebus_dqn_model.keras")
dueling_model = tf.keras.models.load_model("ridebus_dueling_dqn_model.keras")

def evaluate(model, label, episodes=50):
    rewards, corrects, total = [], [], []
    for _ in range(episodes):
        env = RideBusEnv(verbose=False)
        state = env.reset()
        episode_reward = 0
        correct = 0
        steps = 0
        done = False

        while not done:
            state_input = np.expand_dims(state, axis=0).astype(np.float32) / 13.0
            q_values = model.predict(state_input, verbose=0)
            action = np.argmax(q_values[0])
            state, (val_card, val_new), done, info = env.step(action)
            margin = val_new - val_card
            if info["guess_val"] == 0:
                margin = -margin
            reward = np.sign(margin)
            episode_reward += reward
            correct += int(reward == 1)
            steps += 1

        rewards.append(episode_reward)
        corrects.append(correct)
        total.append(steps)
        

    return {
        "label": label,
        "rewards": rewards,
        "correct_guesses": corrects,
        "total_guesses": total
    }

# Evaluate both models
dqn_data = evaluate(standard_model, "Standard DQN")
dueling_data = evaluate(dueling_model, "Dueling DQN")

# Plot reward distributions
plt.figure(figsize=(10, 5))
sns.histplot(dqn_data["rewards"], kde=True, color="blue", label="Standard DQN", stat="density")
sns.histplot(dueling_data["rewards"], kde=True, color="green", label="Dueling DQN", stat="density")
plt.xlabel("Total Reward")
plt.ylabel("Density")
plt.title("Reward Distribution: Standard vs Dueling DQN")
plt.legend()
plt.tight_layout()
plt.show()

# Plot guess accuracy
dqn_acc = np.mean(np.array(dqn_data["correct_guesses"]) / np.array(dqn_data["total_guesses"]))
dueling_acc = np.mean(np.array(dueling_data["correct_guesses"]) / np.array(dueling_data["total_guesses"]))

plt.bar(["Standard DQN", "Dueling DQN"], [dqn_acc, dueling_acc], color=["blue", "green"])
plt.ylabel("Accuracy")
plt.title("Average Guess Accuracy")
plt.tight_layout()
plt.show()

# Basic unpaired t-test using vanilla Python
def t_test_independent(a, b):
    mean_a, mean_b = mean(a), mean(b)
    var_a = np.var(a, ddof=1)  # ddof=1 for sample variance, same as statistics.stdev()**2
    var_b = np.var(b, ddof=1)
    n_a, n_b = len(a), len(b)
    t_stat = (mean_a - mean_b) / sqrt(var_a / n_a + var_b / n_b)
    return t_stat

# Perform test
stat = t_test_independent(dqn_data["rewards"], dueling_data["rewards"])
print(f"\nT-test (manual calc): t={stat:.3f} (not reporting p-value without scipy)")