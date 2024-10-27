import gymnasium as gym
import tictactoe_env
import time
import pygame

from q_learning_agent import QLearningAgent

# Initialize environment and two agents
env = gym.make("tictactoe-v0", render_mode="human")  # Use "human" mode for visual rendering
agent_X = QLearningAgent(env.action_space, exploration_decay=0.9995)

# Training parameters
num_episodes = 100000
render_interval = 5000
win_count_X, draw_count = 0, 0
invalid_count = 0

for episode in range(num_episodes):
    (state, current_player), _ = env.reset()
    if episode % 2 == 0:
        current_player = 1  # Agent starts
    else:
        current_player = -1  # Random opponent starts

    done = False
    episode_reward_X = 0  # Track rewards for each agent

    # Decide whether to render this episode
    render_this_episode = episode % render_interval == 0
    invalid = False
    invalid_attempts = 0
    while not done:
        if render_this_episode:
            env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    pygame.quit()
                    exit()

        # Choose the agent based on the current player
        if current_player == 1:
            action = agent_X.choose_action(str(state))
        else:
            action = env.action_space.sample()

        # Step through the environment
        (next_state, current_player), reward, done, truncated, info = env.step(action)

        # If the move was invalid, heavily penalize the current player
        if "invalid" in info and info["invalid"]:
            invalid_count += 1
            invalid_attempts += 1

            if invalid_attempts > 10:  # Reset the game if stuck in repeated invalid moves
                done = True
                break

            # Penalize the current agent but don't switch players yet
            if current_player == 1:
                agent_X.update_q_value(str(state), action, reward, str(next_state), done)

            continue  # Give the player another chance without switching turns

        # Update Q-tables for each agent based on their role if move was valid
        if current_player == 1:  # Agent X just played
            agent_X.update_q_value(str(state), action, reward, str(next_state), done)
            episode_reward_X += reward

        # Add a delay for visibility
        if render_this_episode:
            time.sleep(0.2)  # Adjust delay as desired

        current_player = -current_player
        state = next_state

    if render_this_episode:
        print(f"Episode {episode} | Reward: {episode_reward_X} | Invalid moves: {invalid_count}")
        env.render()  # Display the final board
        time.sleep(1)  # Add a delay before resetting the environment

    # Track win/loss/draw based on final reward
    if current_player == 1 and reward == 10:  # Agent X wins
        win_count_X += 1
    elif reward == 0:  # Draw
        draw_count += 1

    # Decay exploration rates for both agents
    agent_X.decay_exploration()

    if episode % 1000 == 0:
        print(f"Episode {episode}")
        print(f"Agent X Wins: {win_count_X}, Draws: {draw_count}")

env.close()
