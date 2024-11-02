from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
from gymnasium.spaces import Discrete


@dataclass
class AgentConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 1.0
    exploration_decay: float = 0.9995
    min_exploration_rate: float = 0.01
    initial_q_value: float = 0.01


class QLearningAgent:
    def __init__(self, action_space: Discrete, config: Optional[AgentConfig] = None):
        self.action_space_size = action_space.n
        self.config = config or AgentConfig()
        self.q_table: Dict[str, Tuple[float, ...]] = {}
        self.exploration_rate = self.config.exploration_rate
        self.player_symbol: Optional[str] = None

    def find_available_moves(self, state: str) -> List[int]:
        return [i for i, val in enumerate(state) if val == "-"]

    def initialize_state_values(self, state: str) -> None:
        if state not in self.q_table:
            self.q_table[state] = tuple(
                self.config.initial_q_value for _ in range(self.action_space_size)
            )

    def select_best_move(self, state: str, available_moves: List[int]) -> int:
        q_values = self.q_table[state]
        move_values = [(move, q_values[move]) for move in available_moves]
        highest_value = max(move_values, key=lambda x: x[1] + random.uniform(0, 1e-6))[1]

        threshold = 1e-6
        optimal_moves = [
            move
            for move, value in move_values
            if abs(value - highest_value) < threshold
        ]

        return random.choice(optimal_moves)

    def choose_action(self, state: str) -> int:
        available_moves = self.find_available_moves(state)
        if not available_moves:
            return 0

        self.initialize_state_values(state)

        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(available_moves)

        return self.select_best_move(state, available_moves)

    def update(self, state: str, action: int, reward: float, next_state: str, done: bool) -> None:
        self.initialize_state_values(state)
        self.initialize_state_values(next_state)

        current_value = self.q_table[state][action]

        if done:
            target_value = reward
        else:
            available_moves = self.find_available_moves(next_state)
            future_values = [self.q_table[next_state][move] for move in available_moves]
            max_future_value = max(future_values) if future_values else 0
            target_value = reward + self.config.discount_factor * max_future_value

        updated_value = current_value + self.config.learning_rate * (target_value - current_value)

        state_values = list(self.q_table[state])
        state_values[action] = updated_value
        self.q_table[state] = tuple(state_values)

    def decay_exploration(self) -> None:
        self.exploration_rate = max(
            self.config.min_exploration_rate,
            self.exploration_rate * self.config.exploration_decay,
        )
