import time
import pygame
import random
from q_learning_agent import QLearningAgent, AgentConfig
from tictactoe_env import TicTacToeEnv

env = TicTacToeEnv(render_mode="human")

config = AgentConfig(
    learning_rate=0.2,
    discount_factor=0.99,
    exploration_rate=1.0,
    exploration_decay=0.99995,
    initial_q_value=0.0,
)
agent_X = QLearningAgent(env.action_space, config)

num_episodes = 1_000_000
render_interval = 100_000
win_count_X = 0
win_count_O = 0
draw_count = 0
window_size = 2000

recent_results = []


def get_random_valid_move(board_state):
    valid_moves = [i for i, val in enumerate(board_state) if val == "-"]
    return random.choice(valid_moves) if valid_moves else 0


for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    state = obs["board"]
    last_agent_state = None
    last_agent_action = None

    render_this_episode = episode % render_interval == 0
    current_player = obs["current_player"]

    while not done:
        if render_this_episode:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    pygame.quit()
                    exit()

        if current_player == "X":
            action = agent_X.choose_action(state)
            last_agent_state = state
            last_agent_action = action
        else:
            action = get_random_valid_move(state)

        obs_next, reward, done, truncated, info = env.step(action)
        next_state = obs_next["board"]

        if current_player == "X":
            agent_X.update(state, action, reward, next_state, done)

        if done:
            if reward == 1 and current_player == "X":
                recent_results.append("X")
            elif reward == 1 and current_player == "O":
                recent_results.append("O")
                agent_X.update(
                    last_agent_state, last_agent_action, -reward, next_state, done
                )
            elif reward == 0:
                recent_results.append("D")
                agent_X.update(
                    last_agent_state, last_agent_action, reward, next_state, done
                )

        if render_this_episode:
            time.sleep(0.3)

        current_player = obs_next["current_player"]

        state = next_state

    if render_this_episode:
        env.render()
        time.sleep(1.0)

    if reward > 0:
        if current_player == "X":
            win_count_X += 1
        else:
            win_count_O += 1
    elif reward == 0:
        draw_count += 1

    agent_X.decay_exploration()

    if len(recent_results) > window_size:
        recent_results.pop(0)

    # Print progress
    if episode % 1000 == 0:
        recent_X_wins = recent_results.count("X")
        recent_O_wins = recent_results.count("O")
        recent_draws = recent_results.count("D")

        overall_win_rate = (win_count_X / (episode + 1)) * 100
        recent_win_rate = (recent_X_wins / len(recent_results)) * 100

        print(f"\nEpisode {episode}")
        print(
            f"Overall - Agent X Wins: {win_count_X}, Agent O Wins: {win_count_O}, Draws: {draw_count}"
        )
        print(f"Overall Win Rate: {overall_win_rate:.1f}%")
        print(
            f"Recent Win Rate (last {len(recent_results)} games): {recent_win_rate:.1f}%"
        )
        print(
            f"Recent - X Wins: {recent_X_wins}, O Wins: {recent_O_wins}, Draws: {recent_draws}"
        )
        print(f"X Exploration Rate: {agent_X.exploration_rate:.3f}")


env.close()
