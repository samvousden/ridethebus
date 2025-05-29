import gym
from gym import spaces
import numpy as np

import ride_bus
from ride_bus import RideTheBus

class RideBusEnv(gym.Env):
    def __init__(self, verbose=False):
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

    def step(self, action):
        card_index = action // 2
        guess_val = action % 2
        board_card = self.game.board[card_index][0]
        highlow = "higher" if guess_val == 1 else "lower"

        val_card = self.game.cardvals[board_card]

        prev_deck = self.game.deck.copy()
        correct = self.game.input_guess(board_card, highlow)
        new_card = self.game.board[card_index][0]
        val_new = self.game.cardvals[new_card]

        done = len(self.game.deck) == 0

        if self.verbose:
            print(f"\n🃏 Selected card index {card_index}: {board_card}")
            print(f"🤔 Guessed: {highlow}")
            print(f"🎲 New card drawn: {new_card}")
            print(f"🎯 Guess was {'correct ✅' if correct else 'wrong ❌'}")
            print(f"🏆 Score: {self.game.score}")
            self.render()

        return self._get_obs(), (val_card, val_new), done, {
            "card_index": card_index,
            "card_before": board_card,
            "card_after": new_card,
            "correct": correct,
            "guess_val": guess_val
        }

    def render(self, mode="human"):
        print("\n🎴 Current Board:")
        for i in range(0, 16, 4):
            row = self.game.board[i:i+4]
            print("  ".join(c[0].rjust(2) for c in row))
        print(f"Score: {self.game.score}\n")