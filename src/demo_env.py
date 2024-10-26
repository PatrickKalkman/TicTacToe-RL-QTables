import gymnasium as gym
import tictactoe_env
import numpy as np
import matplotlib.pyplot as plt

# Create the environment with "rgb_array" mode to get numpy arrays
env = gym.make("tictactoe-v0", render_mode="rgb_array")

# Reset the environment for a new game
state, _ = env.reset()
frames = []

done = False
while not done:
    frame = env.render()  # Get the rendered frame as an array
    frames.append(frame)   # Store the frame for later visualization or video
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)

env.close()
