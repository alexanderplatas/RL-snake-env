from stable_baselines3.common.env_checker import check_env
from snakeenv import SnakeEnv

################# Check Environment ###################

env = SnakeEnv()
env.unwrapped.activate_render()

while True:
    check_env(env)
