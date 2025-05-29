# Play a human vs model game of Ride the Bus
import numpy as np
import tensorflow as tf
from ride_bus_env import RideBusEnv
from keras.saving import register_keras_serializable

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

def display_board(board):
    print("\nCurrent board:")
    for i in range(0, 16, 4):
        row = board[i:i+4]
        print("  ".join(card.rjust(2) for card in row))

def get_human_input():
    while True:
        try:
            index = int(input("Select card index (0–15): "))
            if 0 <= index < 16:
                break
        except ValueError:
            pass
        print("Invalid input. Try again.")

    while True:
        guess = input("Guess (higher/lower): ").strip().lower()
        if guess in ["higher", "lower"]:
            break
        print("Invalid guess. Try again.")

    return index, guess

def play_game():
    env = RideBusEnv(verbose=True)
    state = env.reset()

    human_score = 0
    model_score = 0
    human_streak = 0
    model_streak = 0

    while True:
        print("\n==============================")
        display_board([card[0] for card in env.game.board])

        choice = input("Who should play this turn? (human/model/quit): ").strip().lower()
        if choice == "quit":
            break
        elif choice == "model":
            state_input = np.expand_dims(state, axis=0).astype(np.float32) / 13.0
            q_values = dueling_model.predict(state_input, verbose=0)
            action = np.argmax(q_values[0])
            card_index = action // 2
            guess = "higher" if action % 2 == 1 else "lower"
            print(f"Model selects card {card_index} and guesses {guess}")
            player = "model"
        elif choice == "human":
            card_index, guess = get_human_input()
            player = "human"
        else:
            print("Invalid choice. Try again.")
            continue

        board_card = env.game.board[card_index][0]
        val_card = env.game.cardvals[board_card]

        correct = env.game.input_guess(board_card, guess)
        new_card = env.game.board[card_index][0]
        val_new = env.game.cardvals[new_card]
        margin = val_new - val_card
        if guess == "lower":
            margin = -margin
        reward = np.sign(margin)

        print(f"You drew: {new_card} (was {board_card})")
        print(f"Your guess was {'correct ✅' if correct else 'wrong ❌'}")

        # Update streaks and scores
        if player == "human":
            if correct:
                human_streak += 1
                if human_streak == 3:
                    human_score += 1
                    human_streak = 0
                    print("🏆 Human scored a point!")
            else:
                human_streak = 0
        elif player == "model":
            if correct:
                model_streak += 1
                if model_streak == 3:
                    model_score += 1
                    model_streak = 0
                    print("🤖 Model scored a point!")
            else:
                model_streak = 0

        print(f"Scores — Human: {human_score}, Model: {model_score}")

        state = np.array([env.game.cardvals[c[0]] for c in env.game.board], dtype=np.int32)

        if len(env.game.deck) == 0:
            print("Game over! No more cards in deck.")
            print(f"Final Scores — Human: {human_score}, Model: {model_score}")
            break

if __name__ == "__main__":
    play_game()
