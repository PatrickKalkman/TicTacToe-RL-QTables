# TicTacToe Reinforcement Learning with Q-Learning

![Cover Selfplay](cover_selfplay.jpg)

This repository contains the code for a Q-learning implementation of Tic-Tac-Toe, demonstrating fundamental principles of reinforcement learning through self-play and training against random opponents.

Read the full article: [How Did AlphaGo Beat Lee Sedol? From AlphaGo to Tic-Tac-Toe: Building Your First AI Game Player](https://medium.com/ai-advances/how-did-alphago-beat-lee-sedol-1a160d76612b)

## Features

- Complete Tic-Tac-Toe environment using OpenAI's Gymnasium
- Q-learning agent implementation
- Two training modes:
  - Training against a random opponent
  - Self-play training with two Q-learning agents
- Progress tracking and visualization
- PyGame-based game rendering

## Requirements

- Python 3.12
- Poetry package manager
- Standard CPU (no GPU needed)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PatrickKalkman/TicTacToe-RL-Q
```

2. Set up the environment:
```bash
poetry install
poetry shell
```

3. Run the training scripts:
```bash
cd src
python train_agent_vs_random.py  # Train against random opponent
python train_agent_vs_agent.py   # Train using self-play
```

## Project Structure

- `src/`
  - `train_agent_vs_random.py` - Training script for single agent vs random opponent
  - `train_agent_vs_agent.py` - Training script for self-play between two agents

## Training Strategies

### Against Random Player
- Agent learns basic winning strategies
- Typically achieves 75-80% win rate after training
- Shows rapid initial improvement followed by gradual refinement

### Self-Play Training
- Two agents learn simultaneously
- Results in more sophisticated strategies
- Higher percentage of draws as agents improve
- More closely mimics human expert play

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License