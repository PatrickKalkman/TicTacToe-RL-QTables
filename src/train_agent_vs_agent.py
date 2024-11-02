from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import pygame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from q_learning_agent import QLearningAgent, AgentConfig
from tictactoe_env import TicTacToeEnv


@dataclass
class TrainingConfig:
    episodes: int = 1_000_000
    render_interval: int = 200_000
    stats_interval: int = 1_000
    render_delay: float = 0.1
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

        # Lists to store data for plotting
        self.episodes: List[int] = []
        self.x_winrates: List[float] = []
        self.o_winrates: List[float] = []
        self.draw_rates: List[float] = []
        self.x_exploration_rates: List[float] = []
        self.o_exploration_rates: List[float] = []

    def update(self, reward: float, current_player: str, episode: int,
               agent_x: QLearningAgent, agent_o: QLearningAgent) -> None:
        if reward > 0:
            if current_player == "X":
                self.wins_x += 1
            else:
                self.wins_o += 1
        elif reward == 0:
            self.draws += 1

        # Store data for plotting every N episodes
        if episode % 1000 == 0:  # Adjust frequency as needed
            total_games = self.wins_x + self.wins_o + self.draws
            self.episodes.append(episode)
            self.x_winrates.append((self.wins_x / total_games) * 100)
            self.o_winrates.append((self.wins_o / total_games) * 100)
            self.draw_rates.append((self.draws / total_games) * 100)
            self.x_exploration_rates.append(agent_x.exploration_rate)
            self.o_exploration_rates.append(agent_o.exploration_rate)

    def print_progress(
        self, episode: int, agent_x: QLearningAgent, agent_o: QLearningAgent
    ) -> None:
        print(f"\nEpisode {episode}")
        print(
            f"Agent X Wins: {self.wins_x}, Agent O Wins: {self.wins_o}, Draws: {self.draws}"
        )
        print(f"X Exploration Rate: {agent_x.exploration_rate:.3f}")
        print(f"O Exploration Rate: {agent_o.exploration_rate:.3f}")

    def plot_training_progress(self):
        # Set style
        sns.set_style("whitegrid")
        plt.ion()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor('#f0f0f0')

        # Setup all axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('white')
            ax.spines['bottom'].set_color('#666666')
            ax.spines['top'].set_color('#666666')
            ax.spines['right'].set_color('#666666')
            ax.spines['left'].set_color('#666666')
            ax.tick_params(colors='#666666')

        # Plot 1: Win rates and draw rate
        ax1.plot(self.episodes, self.x_winrates, label='Agent X Wins', color='#3498db')
        ax1.plot(self.episodes, self.o_winrates, label='Agent O Wins', color='#e74c3c')
        ax1.plot(self.episodes, self.draw_rates, label='Draws', color='#95a5a6')
        ax1.set_title('Win and Draw Rates Over Time', pad=15, fontsize=12, color='#444444')
        ax1.set_xlabel('Episodes', fontsize=10, color='#666666')
        ax1.set_ylabel('Rate (%)', fontsize=10, color='#666666')
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot 2: Exploration rates
        ax2.plot(self.episodes, self.x_exploration_rates,
                 label='Agent X', color='#3498db')
        ax2.plot(self.episodes, self.o_exploration_rates,
                 label='Agent O', color='#e74c3c')
        ax2.set_title('Exploration Rates Over Time', pad=15, fontsize=12, color='#444444')
        ax2.set_xlabel('Episodes', fontsize=10, color='#666666')
        ax2.set_ylabel('Exploration Rate', fontsize=10, color='#666666')
        ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Plot 3: Relative performance
        performance_ratio = np.array(self.x_winrates) / np.where(np.array(self.o_winrates) == 0, 1, np.array(self.o_winrates))
        ax3.plot(self.episodes, performance_ratio, color='#9b59b6')
        ax3.axhline(y=1.0, color='#666666', linestyle='--')
        ax3.set_title('Relative Performance (X/O Win Ratio)', pad=15, fontsize=12, color='#444444')
        ax3.set_xlabel('Episodes', fontsize=10, color='#666666')
        ax3.set_ylabel('Ratio', fontsize=10, color='#666666')
        ax3.grid(True, linestyle='--', alpha=0.7)

        # Plot 4: Game outcomes distribution
        labels = ['X Wins', 'O Wins', 'Draws']
        counts = [self.wins_x, self.wins_o, self.draws]
        colors = ['#3498db', '#e74c3c', '#95a5a6']
        bars = ax4.bar(labels, counts, color=colors)
        ax4.set_title('Total Game Outcomes', pad=15, fontsize=12, color='#444444')
        ax4.set_ylabel('Number of Games', fontsize=10, color='#666666')
        ax4.grid(True, linestyle='--', alpha=0.7, axis='y')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)


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

        # Initialize matplotlib plot
        sns.set_style("whitegrid")
        plt.ion()
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.setup_plots()

    def setup_plots(self):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('white')
            ax.spines['bottom'].set_color('#666666')
            ax.spines['top'].set_color('#666666')
            ax.spines['right'].set_color('#666666')
            ax.spines['left'].set_color('#666666')
            ax.tick_params(colors='#666666')

    def train(self) -> None:
        for episode in range(self.config.episodes):
            self.run_episode(episode)

        self.stats.plot_training_progress()

    def update_plots(self, episode: int):
        if episode % self.config.stats_interval != 0:
            return

        total_games = self.stats.wins_x + self.stats.wins_o + self.stats.draws

        # Store data for plotting
        self.stats.episodes.append(episode)
        self.stats.x_winrates.append((self.stats.wins_x / total_games) * 100)
        self.stats.o_winrates.append((self.stats.wins_o / total_games) * 100)
        self.stats.draw_rates.append((self.stats.draws / total_games) * 100)
        self.stats.x_exploration_rates.append(self.agents["X"].exploration_rate)
        self.stats.o_exploration_rates.append(self.agents["O"].exploration_rate)

        # Clear all plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # Plot 1: Win rates and draw rate
        self.ax1.plot(self.stats.episodes, self.stats.x_winrates, 
                      label='Agent X Wins', color='#3498db')
        self.ax1.plot(self.stats.episodes, self.stats.o_winrates, 
                      label='Agent O Wins', color='#e74c3c')
        self.ax1.plot(self.stats.episodes, self.stats.draw_rates, 
                      label='Draws', color='#95a5a6')
        self.ax1.set_title('Win and Draw Rates Over Time', pad=15, fontsize=12, color='#444444')
        self.ax1.set_xlabel('Episodes', fontsize=10, color='#666666')
        self.ax1.set_ylabel('Rate (%)', fontsize=10, color='#666666')
        self.ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
        self.ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot 2: Exploration rates
        self.ax2.plot(self.stats.episodes, self.stats.x_exploration_rates, 
                      label='Agent X', color='#3498db')
        self.ax2.plot(self.stats.episodes, self.stats.o_exploration_rates, 
                      label='Agent O', color='#e74c3c')
        self.ax2.set_title('Exploration Rates Over Time', pad=15, fontsize=12, color='#444444')
        self.ax2.set_xlabel('Episodes', fontsize=10, color='#666666')
        self.ax2.set_ylabel('Exploration Rate', fontsize=10, color='#666666')
        self.ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
        self.ax2.grid(True, linestyle='--', alpha=0.7)

        # Plot 3: Relative performance
        if len(self.stats.o_winrates) > 0:
            performance_ratio = np.array(self.stats.x_winrates) / np.where(
                np.array(self.stats.o_winrates) == 0, 1, np.array(self.stats.o_winrates)
            )
            self.ax3.plot(self.stats.episodes, performance_ratio, color='#9b59b6')
            self.ax3.axhline(y=1.0, color='#666666', linestyle='--')

        self.ax3.set_title('Relative Performance (X/O Win Ratio)', pad=15, fontsize=12, color='#444444')
        self.ax3.set_xlabel('Episodes', fontsize=10, color='#666666')
        self.ax3.set_ylabel('Ratio', fontsize=10, color='#666666')
        self.ax3.grid(True, linestyle='--', alpha=0.7)

        # Plot 4: Game outcomes distribution
        labels = ['X Wins', 'O Wins', 'Draws']
        counts = [self.stats.wins_x, self.stats.wins_o, self.stats.draws]
        colors = ['#3498db', '#e74c3c', '#95a5a6']
        bars = self.ax4.bar(labels, counts, color=colors)
        self.ax4.set_title('Total Game Outcomes', pad=15, fontsize=12, color='#444444')
        self.ax4.set_ylabel('Number of Games', fontsize=10, color='#666666')
        self.ax4.grid(True, linestyle='--', alpha=0.7, axis='y')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax4.text(bar.get_x() + bar.get_width() / 2., height,
                          f'{int(height)}',
                          ha='center', va='bottom')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

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

        self.stats.update(reward, current_player, episode, self.agents["X"], self.agents["O"])

        for agent in self.agents.values():
            agent.decay_exploration()

        if episode % self.config.stats_interval == 0:
            self.stats.print_progress(episode, self.agents["X"], self.agents["O"])
            self.update_plots(episode)

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
        learning_rate=0.2,
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
        plt.show()
        plt.savefig('training_plot.png')
    finally:
        env.close()


if __name__ == "__main__":
    main()
