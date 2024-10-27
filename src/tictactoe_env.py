import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from pygame_renderer import PygameRenderer


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        super(TicTacToeEnv, self).__init__()
        self.renderer = PygameRenderer(window_size=300)
        self.render_mode = render_mode

        # Observation space includes the board and the current player
        self.observation_space = spaces.Tuple((
            spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8),
            spaces.Discrete(2)  # 0 for -1 (O) and 1 for 1 (X)
        ))
        self.action_space = spaces.Discrete(9)

        self.state = np.zeros(9, dtype=np.int8)
        self.current_player = 1
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(9, dtype=np.int8)
        self.current_player = random.choice([1, -1])  # Randomly choose between player 1 (X) and player -1 (O)
        self.done = False
        return (self.state, self.current_player), {}

    def step(self, action):
        if self.state[action] != 0:
            return (self.state, self.current_player), -5, False, False, {"invalid": True}

        # Place the current player's mark and check the game state
        self.state[action] = self.current_player
        reward, self.done = self._check_game_status()

        # Switch players if game continues
        if not self.done:
            self.current_player = -self.current_player

        return (self.state, self.current_player), reward, self.done, False, {}

    def _check_game_status(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]

        for combo in winning_combinations:
            if abs(self.state[combo[0]] + self.state[combo[1]] + self.state[combo[2]]) == 3:
                return 10, True

        # Check for a draw
        if 0 not in self.state:
            return 0, True  # Draw

        return -1, False

    def render(self):
        if self.render_mode == "human":
            self.renderer.draw_board(self.state)
        elif self.render_mode == "rgb_array":
            self.renderer.draw_board(self.state)
            return pygame.surfarray.array3d(self.renderer.screen).transpose(1, 0, 2)

    def close(self):
        self.renderer.close()


from gymnasium.envs.registration import register

register(
    id='tictactoe-v0',
    entry_point='tictactoe_env:TicTacToeEnv',
)
