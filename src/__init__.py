from gymnasium.envs.registration import register

__version__ = "0.1.0"

register(
    id="TicTacToe-v0",
    entry_point="tictactoe_env.env:TicTacToeEnv",
    max_episode_steps=9,
)
