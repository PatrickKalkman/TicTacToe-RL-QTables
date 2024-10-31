from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import pygame
import random

from q_learning_agent import QLearningAgent, AgentConfig
from tictactoe_env import TicTacToeEnv


@dataclass
class TrainingConfig:
    episodes: int = 1_000_000
    render_interval: int = 100_000
    stats_interval: int = 1_000
    window_size: int = 1_000
    render_delay: float = 0.3
    end_episode_delay: float = 1.0


class TrainingStats:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.recent_results: List[str] = []
        self.wins_x = 0
        self.wins_o = 0
        self.draws = 0

    def update(self, winner: str) -> None:
        self.recent_results.append(winner)
        if len(self.recent_results) > self.window_size:
            self.recent_results.pop(0)

        if winner == "X":
            self.wins_x += 1
        elif winner == "O":
            self.wins_o += 1
        else:
            self.draws += 1

    def get_stats(self, episode: int) -> Tuple[float, float]:
        recent_x_wins = self.recent_results.count("X")
        total_episodes = episode + 1

        overall_winrate = (self.wins_x / total_episodes) * 100
        recent_winrate = (recent_x_wins / len(self.recent_results)) * 100

        return overall_winrate, recent_winrate

    def get_recent_counts(self) -> Tuple[int, int, int]:
        return (
            self.recent_results.count("X"),
            self.recent_results.count("O"),
            self.recent_results.count("D"),
        )


class TicTacToeTrainer:
    def __init__(
        self, env: TicTacToeEnv, agent: QLearningAgent, config: TrainingConfig
    ):
        self.env = env
        self.agent = agent
        self.config = config
        self.stats = TrainingStats(config.window_size)

    def select_action(
        self, state: str, current_player: str
    ) -> Tuple[int, Optional[str], Optional[int]]:
        if current_player == "X":
            action = self.agent.choose_action(state)
            return action, state, action
        else:
            return self.select_random_move(state), None, None

    def select_random_move(self, board_state: str) -> int:
        valid_moves = [i for i, val in enumerate(board_state) if val == "-"]
        return random.choice(valid_moves) if valid_moves else 0

    def handle_episode_end(
        self,
        reward: float,
        current_player: str,
        last_state: Optional[str],
        last_action: Optional[int],
        next_state: str,
    ) -> None:
        if reward == 1 and current_player == "X":
            self.stats.update("X")
        elif reward == 1 and current_player == "O":
            self.stats.update("O")
            if last_state and last_action is not None:
                self.agent.update(last_state, last_action, -reward, next_state, True)
        else:
            self.stats.update("D")
            if last_state and last_action is not None:
                self.agent.update(last_state, last_action, reward, next_state, True)

    def print_progress(self, episode: int) -> None:
        if episode % self.config.stats_interval != 0:
            return

        overall_wr, recent_wr = self.stats.get_stats(episode)
        recent_x, recent_o, recent_d = self.stats.get_recent_counts()

        print(f"\nEpisode {episode}")
        print(
            f"Overall - X W: {self.stats.wins_x}, O W: {self.stats.wins_o}, Dr: {self.stats.draws}"
        )
        print(f"Overall Win Rate: {overall_wr:.1f}%")
        print(f"Recent Win (last {len(self.stats.recent_results)}): {recent_wr:.1f}%")
        print(f"Recent - X W: {recent_x}, O W: {recent_o}, Dr: {recent_d}")
        print(f"X Exploration Rate: {self.agent.exploration_rate:.3f}")

    def train(self) -> None:
        for episode in range(self.config.episodes):
            self.run_episode(episode)

    def run_episode(self, episode: int) -> None:
        obs, _ = self.env.reset()
        state = obs["board"]
        current_player = obs["current_player"]
        last_state = last_action = None
        done = False

        render_this_episode = episode % self.config.render_interval == 0

        while not done:
            if render_this_episode:
                self.render_and_check_quit()

            action, last_state, last_action = self.select_action(state, current_player)
            obs_next, reward, done, _, _ = self.env.step(action)
            next_state = obs_next["board"]

            if current_player == "X":
                self.agent.update(state, action, reward, next_state, done)

            if done:
                self.handle_episode_end(
                    reward, current_player, last_state, last_action, next_state
                )
                if render_this_episode:
                    self.env.render()
                    time.sleep(self.config.end_episode_delay)
                break

            if render_this_episode:
                time.sleep(self.config.render_delay)

            current_player = obs_next["current_player"]
            state = next_state

        self.agent.decay_exploration()
        self.print_progress(episode)

    def render_and_check_quit(self) -> None:
        self.env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()


def main():
    env = TicTacToeEnv(render_mode="human")

    agent_config = AgentConfig(
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.99995,
        initial_q_value=0.0,
    )

    training_config = TrainingConfig()

    agent = QLearningAgent(env.action_space, agent_config)
    trainer = TicTacToeTrainer(env, agent, training_config)

    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
