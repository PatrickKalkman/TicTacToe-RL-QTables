from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple, Optional
import random

import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame_renderer import PygameRenderer


class Player(str, Enum):
    X_SYMBOL = "X"
    O_SYMBOL = "O"
    EMPTY = "-"


@dataclass
class GameState:
    board: str
    current_player: Player
    done: bool = False


class WinningPatterns:
    ROWS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    COLUMNS = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
    DIAGONALS = [(0, 4, 8), (2, 4, 6)]

    @classmethod
    def all_patterns(cls) -> List[Tuple[int, int, int]]:
        return cls.ROWS + cls.COLUMNS + cls.DIAGONALS


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Literal["human", "rgb_array"] = "human"):
        super().__init__()
        self.renderer = PygameRenderer()
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "board": spaces.Text(min_length=9, max_length=9, charset="XO-"),
            "current_player": spaces.Discrete(2)})
        self.action_space = spaces.Discrete(9)

        self.state = self._create_initial_state()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict]:
        super().reset(seed=seed)
        self.state = self._create_initial_state()
        self.state.current_player = random.choice([Player.X_SYMBOL, Player.O_SYMBOL])
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        if self._is_valid_move(action):
            self._make_move(action)
            reward, self.state.done = self._evaluate_game_state()
            if not self.state.done:
                self._switch_player()
            return self._get_observation(), reward, self.state.done, False, {}
        return self._get_observation(), -1, True, False, {}

    def _create_initial_state(self) -> GameState:
        return GameState(board=Player.EMPTY * 9, current_player=Player.X_SYMBOL, done=False)

    def _is_valid_move(self, action: int) -> bool:
        return list(self.state.board)[action] == Player.EMPTY

    def _make_move(self, action: int) -> None:
        board_list = list(self.state.board)
        board_list[action] = self.state.current_player
        self.state.board = "".join(board_list)

    def _switch_player(self) -> None:
        self.state.current_player = (
            Player.O_SYMBOL
            if self.state.current_player == Player.X_SYMBOL
            else Player.X_SYMBOL
        )

    def _evaluate_game_state(self) -> Tuple[float, bool]:
        for combo in WinningPatterns.all_patterns():
            line = "".join(self.state.board[i] for i in combo)
            if line in ("XXX", "OOO"):
                return 1, True

        if Player.EMPTY not in self.state.board:
            return 0, True

        return 0, False

    def _get_observation(self) -> Dict[str, Any]:
        return {"board": self.state.board,
                "current_player": self.state.current_player}

    def render(self) -> Optional[Any]:
        self.renderer.render_game_state(self.state.board)
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.renderer.screen).transpose(1, 0, 2)
        return None

    def close(self) -> None:
        self.renderer.close()


from gymnasium.envs.registration import register
register(id="tictactoe-v0", entry_point="tictactoe_env:TicTacToeEnv")
