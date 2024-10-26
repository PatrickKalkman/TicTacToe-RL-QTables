import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from pygame_renderer import PygameRenderer


class TicTacToeEnv(gym.Env):
    """Custom TicTacToe Environment following OpenAI Gymnasium structure"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        super(TicTacToeEnv, self).__init__()
        self.renderer = PygameRenderer(window_size=300)
        self.render_mode = render_mode

        # Define the observation and action spaces
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)
        self.action_space = spaces.Discrete(9)

        self.state = np.zeros(9, dtype=np.int8)
        self.done = False
        self.current_player = 1

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        self.state = np.zeros(9, dtype=np.int8)
        self.done = False
        self.current_player = 1
        return self.state, {}

    def step(self, action):
        if self.state[action] != 0:
            return self.state, -10, True, False, {"invalid": True}

        self.state[action] = self.current_player
        reward, self.done = self._check_game_status()
        self.current_player = -self.current_player

        return self.state, reward, self.done, False, {}

    def render(self):
        """Render the environment based on the selected mode."""
        if self.render_mode == "human":
            self.renderer.draw_board(self.state)
        elif self.render_mode == "rgb_array":
            self.renderer.draw_board(self.state)
            return pygame.surfarray.array3d(self.renderer.screen).transpose(1, 0, 2)

    def _check_game_status(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]

        for combo in winning_combinations:
            if abs(self.state[combo[0]] + self.state[combo[1]] + self.state[combo[2]]) == 3:
                return 1 if self.current_player == 1 else -1, True  # Winning reward

        if 0 not in self.state:
            return 0, True

        return 0, False

    def close(self):
        """Close the pygame renderer."""
        self.renderer.close()


from gymnasium.envs.registration import register

register(
    id='tictactoe-v0',
    entry_point='tictactoe_env:TicTacToeEnv',
)
