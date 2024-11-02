from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import pygame
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
    plot_interval: int = 10_000


class TrainingStats:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.recent_results: List[str] = []
        self.wins_x = 0
        self.wins_o = 0
        self.draws = 0

        # Lists to store data for plotting
        self.episodes: List[int] = []
        self.overall_winrates: List[float] = []
        self.recent_winrates: List[float] = []
        self.exploration_rates: List[float] = []

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

        # Only calculate rates if we have enough games
        if len(self.recent_results) < self.window_size:
            recent_winrate = 0.0
        else:
            recent_winrate = (recent_x_wins / len(self.recent_results)) * 100

        overall_winrate = (self.wins_x / total_episodes) * 100
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

        # Initialize matplotlib plot with style
        sns.set_style("whitegrid")  # Use seaborn's whitegrid style
        plt.ion()  # Enable interactive mode
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.patch.set_facecolor('#f0f0f0')  # Light gray background
        self.setup_plots()

    def setup_plots(self):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('white')
            ax.spines['bottom'].set_color('#666666')
            ax.spines['top'].set_color('#666666')
            ax.spines['right'].set_color('#666666')
            ax.spines['left'].set_color('#666666')
            ax.tick_params(colors='#666666')

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

        # Update plots
        self.update_plots(episode)

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

    def update_plots(self, episode: int):
        if episode % self.config.plot_interval != 0:
            return

        if episode < self.config.window_size:
            return

        overall_wr, recent_wr = self.stats.get_stats(episode)
        recent_x, recent_o, recent_d = self.stats.get_recent_counts()

        # Store data for plotting
        self.stats.episodes.append(episode)
        self.stats.overall_winrates.append(overall_wr)
        self.stats.recent_winrates.append(recent_wr)
        self.stats.exploration_rates.append(self.agent.exploration_rate)
        
        # Clear all plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # Plot 1: Win rates
        self.ax1.plot(self.stats.episodes, self.stats.overall_winrates, 
                    label='Overall Win Rate', color='#2ecc71', linewidth=2)
        self.ax1.plot(self.stats.episodes, self.stats.recent_winrates,
                    label=f'Recent Win Rate (Window={self.config.window_size})',
                    color='#3498db', linewidth=2)
        self.ax1.set_ylim([0, 100])
        self.ax1.set_xlabel('Episodes', fontsize=10, color='#666666')
        self.ax1.set_ylabel('Win Rate (%)', fontsize=10, color='#666666')
        self.ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        self.ax1.set_title('Win Rates Over Time', pad=15, fontsize=12, color='#444444')

        # Plot 2: Exploration rate
        self.ax2.plot(self.stats.episodes, self.stats.exploration_rates, 
                    color='#e74c3c', linewidth=2, label='Exploration Rate')
        self.ax2.set_ylim([0, 1])
        self.ax2.set_xlabel('Episodes', fontsize=10, color='#666666')
        self.ax2.set_ylabel('Rate', fontsize=10, color='#666666')
        self.ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        self.ax2.set_title('Exploration Rate Over Time', pad=15, fontsize=12, color='#444444')

        # Plot 3: Game outcomes distribution
        outcomes = ['X Wins', 'O Wins', 'Draws']
        counts = [recent_x, recent_o, recent_d]
        colors = ['#3498db', '#e74c3c', '#95a5a6']
        bars = self.ax3.bar(outcomes, counts, color=colors)
        self.ax3.set_title(f'Recent Game Outcomes\n(Last {self.config.window_size} games)', 
                        pad=15, fontsize=12, color='#444444')
        self.ax3.set_ylabel('Count', fontsize=10, color='#666666')
        self.ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')

        # Plot 4: Moving average of rewards
        window_size = min(100, len(self.stats.overall_winrates))
        if window_size > 0:
            moving_avg = [sum(self.stats.overall_winrates[max(0, i-window_size):i])/
                        min(i, window_size) for i in range(1, len(self.stats.overall_winrates)+1)]
            self.ax4.plot(self.stats.episodes, moving_avg, 
                        color='#9b59b6', linewidth=2, 
                        label=f'Moving Avg (Window={window_size})')
            self.ax4.set_xlabel('Episodes', fontsize=10, color='#666666')
            self.ax4.set_ylabel('Average Win Rate', fontsize=10, color='#666666')
            self.ax4.legend(frameon=True, facecolor='white', framealpha=0.9)
            self.ax4.grid(True, linestyle='--', alpha=0.7)
            self.ax4.set_title('Moving Average Win Rate', pad=15, fontsize=12, color='#444444')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

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
