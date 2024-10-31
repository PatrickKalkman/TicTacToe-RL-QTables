from dataclasses import dataclass
from typing import Dict, Optional
import time
import pygame

from q_learning_agent import QLearningAgent, AgentConfig
from tictactoe_env import TicTacToeEnv


@dataclass
class TrainingConfig:
    episodes: int = 500_000
    render_interval: int = 100_000
    stats_interval: int = 1_000
    render_delay: float = 0.5
    end_episode_delay: float = 1.0


class GameState:
    def __init__(self, state: str, action: Optional[int] = None):
        self.board = state
        self.action = action


class TrainingStats:
    def __init__(self):
        self.wins_x = 0
        self.wins_o = 0
        self.draws = 0

    def update(self, reward: float, current_player: str) -> None:
        if reward > 0:
            if current_player == "X":
                self.wins_x += 1
            else:
                self.wins_o += 1
        elif reward == 0:
            self.draws += 1

    def print_progress(
        self, episode: int, agent_x: QLearningAgent, agent_o: QLearningAgent
    ) -> None:
        print(f"\nEpisode {episode}")
        print(
            f"Agent X Wins: {self.wins_x}, Agent O Wins: {self.wins_o}, Draws: {self.draws}"
        )
        print(f"X Exploration Rate: {agent_x.exploration_rate:.3f}")
        print(f"O Exploration Rate: {agent_o.exploration_rate:.3f}")


class DualAgentTrainer:
    def __init__(
        self,
        env: TicTacToeEnv,
        agent_x: QLearningAgent,
        agent_o: QLearningAgent,
        config: TrainingConfig,
    ):
        self.env = env
        self.agents: Dict[str, QLearningAgent] = {"X": agent_x, "O": agent_o}
        self.config = config
        self.stats = TrainingStats()

    def train(self) -> None:
        for episode in range(self.config.episodes):
            self.run_episode(episode)

    def run_episode(self, episode: int) -> None:
        obs, _ = self.env.reset()
        state = obs["board"]
        current_player = obs["current_player"]
        last_moves = {"X": GameState(state), "O": GameState(state)}
        done = False

        render_this_episode = episode % self.config.render_interval == 0

        while not done:
            if render_this_episode:
                self.render_and_check_quit()

            current_agent = self.agents[current_player]
            action = current_agent.choose_action(state)

            last_moves[current_player] = GameState(state, action)

            obs_next, reward, done, _, _ = self.env.step(action)
            next_state = obs_next["board"]

            self.update_agents(
                current_player, last_moves, state, action, reward, next_state, done
            )

            if render_this_episode:
                time.sleep(self.config.render_delay)

            current_player = obs_next["current_player"]
            state = next_state

        if render_this_episode:
            self.env.render()
            time.sleep(self.config.end_episode_delay)

        self.stats.update(reward, current_player)

        for agent in self.agents.values():
            agent.decay_exploration()

        if episode % self.config.stats_interval == 0:
            self.stats.print_progress(episode, self.agents["X"], self.agents["O"])

    def update_agents(
        self,
        current_player: str,
        last_moves: Dict[str, GameState],
        state: str,
        action: int,
        reward: float,
        next_state: str,
        done: bool,
    ) -> None:
        current_agent = self.agents[current_player]
        other_player = "O" if current_player == "X" else "X"
        other_agent = self.agents[other_player]

        # Update current agent
        current_agent.update(state, action, reward, next_state, done)

        # If game is over with a winner, update the other agent
        if done and reward != 0:
            other_last_move = last_moves[other_player]
            if other_last_move.action is not None:
                other_agent.update(
                    other_last_move.board,
                    other_last_move.action,
                    -reward,
                    next_state,
                    done,
                )

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
        exploration_decay=0.999995,
        initial_q_value=0.0,
    )

    agent_x = QLearningAgent(env.action_space, agent_config)
    agent_o = QLearningAgent(env.action_space, agent_config)

    training_config = TrainingConfig()
    trainer = DualAgentTrainer(env, agent_x, agent_o, training_config)

    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
