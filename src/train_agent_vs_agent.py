import time
import pygame
from q_learning_agent import QLearningAgent, AgentConfig
from tictactoe_env import TicTacToeEnv

# Initialize environment and two agents
env = TicTacToeEnv(render_mode="human")

config = AgentConfig(
    learning_rate=0.2,
    discount_factor=0.99,
    exploration_rate=1.0,
    exploration_decay=0.99995,
    initial_q_value=0.0,
)
agent_X = QLearningAgent(env.action_space, config)
agent_O = QLearningAgent(env.action_space, config)

num_episodes = 600_000
render_interval = 100_000
win_count_X = 0
win_count_O = 0
draw_count = 0

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    state = obs["board"]
    last_agent_X_state = None
    last_agent_X_action = None
    last_agent_O_state = None
    last_agent_O_action = None

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

        current_agent = agent_X if current_player == "X" else agent_O

        action = current_agent.choose_action(state)
        if current_player == "X":
            last_agent_X_state = state
            last_agent_X_action = action
        else:
            last_agent_O_state = state
            last_agent_O_action = action

        obs_next, reward, done, truncated, info = env.step(action)
        next_state = obs_next["board"]

        if current_player == "X":
            agent_X.update(state, action, reward, next_state, done)
        else:  # O's turn
            agent_O.update(state, action, reward, next_state, done)

        # Handle game end
        if done and reward != 0:
            if current_player == "X":
                agent_X.update(state, action, reward, next_state, done)
                agent_O.update(
                    last_agent_O_state, last_agent_O_action, -reward, next_state, done
                )
            else:
                agent_O.update(state, action, -reward, next_state, done)
                agent_X.update(
                    last_agent_X_state, last_agent_X_action, reward, next_state, done
                )

        if render_this_episode:
            time.sleep(0.5)

        current_player = obs_next["current_player"]

        state = next_state

    if render_this_episode:
        env.render()
        time.sleep(1.0)

    if reward > 0:  # Win
        if current_player == "X":
            win_count_X += 1
        else:
            win_count_O += 1
    elif reward == 0:  # Draw
        draw_count += 1

    agent_X.decay_exploration()
    agent_O.decay_exploration()

    # Print progress
    if episode % 1000 == 0:
        print(f"\nEpisode {episode}")
        print(
            f"Agent X Wins: {win_count_X}, Agent O Wins: {win_count_O}, Draws: {draw_count}"
        )
        print(f"X Exploration Rate: {agent_X.exploration_rate:.3f}")
        print(f"O Exploration Rate: {agent_O.exploration_rate:.3f}")

agent_X.print_board_values()
agent_O.print_board_values()

env.close()
