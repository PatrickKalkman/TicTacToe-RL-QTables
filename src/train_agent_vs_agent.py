import gymnasium as gym
import tictactoe_env
import time
import pygame

from q_learning_agent import QLearningAgent

# Initialize environment and two agents
env = gym.make("tictactoe-v0", render_mode="human")  # Use "human" mode for visual rendering
agent_X = QLearningAgent(env.action_space, exploration_decay=0.999)
agent_O = QLearningAgent(env.action_space, exploration_decay=0.999)

# Training parameters
num_episodes = 500000
render_interval = 5000
win_count_X = 0
win_count_O = 0
draw_count = 0
invalid_count = 0

for episode in range(num_episodes):
    (state, current_player), _ = env.reset(seed=episode)

    done = False
    episode_reward_X = 0
    episode_reward_O = 0

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
            action = agent_O.choose_action(str(state))

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
            else:
                agent_O.update_q_value(str(state), action, reward, str(next_state), done)

            continue  # Give the player another chance without switching turns

        # Update Q-tables for each agent based on their role if move was valid
        if current_player == 1:  # Agent X just played
            agent_X.update_q_value(str(state), action, reward, str(next_state), done)
            episode_reward_X += reward
        else:
            agent_O.update_q_value(str(state), action, reward, str(next_state), done)
            episode_reward_O += reward

        # Add a delay for visibility
        if render_this_episode:
            time.sleep(0.1)  # Adjust delay as desired

        state = next_state

    if render_this_episode:
        print(f"Episode {episode} | Reward X: {episode_reward_X} | Invalid moves: {invalid_count}")
        print(f"Episode {episode} | Reward O: {episode_reward_O} | Invalid moves: {invalid_count}")
        env.render()  # Display the final board
        time.sleep(0.5)  # Add a delay before resetting the environment

    # Track win/loss/draw based on final reward
    if reward == 10:
        if current_player == 1:
            win_count_X += 1
        else:
            win_count_O += 1
    elif reward == 0:  # Draw
        draw_count += 1

    # Decay exploration rates for both agents
    agent_X.decay_exploration()
    agent_O.decay_exploration()

    if episode % 5000 == 0:
        print(f"Episode {episode}")
        print(f"Agent X Wins: {win_count_X}, Agent O Wins: {win_count_O} Draws: {draw_count}")
        agent_X.render_q_table()

env.close()
