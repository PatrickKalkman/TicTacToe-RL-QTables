import numpy as np
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.2, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.9999):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = defaultdict(lambda: np.random.uniform(-0.01, 0.01, action_space.n)) 

    def choose_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if random.uniform(0, 1) < self.exploration_rate:
            return self.action_space.sample()  # Explore
        else:
            q_values = self.q_table[state]
            max_q_value = np.max(q_values)
            best_actions = [action for action, value in enumerate(q_values) if value == max_q_value]
            return random.choice(best_actions)  # Randomly pick among the best

    def update_q_value(self, state, action, reward, next_state, done=False):
        """Update Q-value based on the action taken and received reward."""
        if done:
            td_target = reward
        else:
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_delta

    def decay_exploration(self, min_exploration_rate=0.1):
        """Decay the exploration rate after each episode."""
        self.exploration_rate = max(min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def render_q_table(self):
        """Print the Q-table for debugging purposes."""
        for state, action_values in self.q_table.items():
            print(f"State: {state}")
            for action, value in enumerate(action_values):
                print(f"  Action {action}: {value}")
