from gymnasium.wrappers import TransformObservation
from stable_baselines3 import PPO, DQN, A2C
from test_model import evaluate_policy
from snakeenv import SnakeEnv
from snakeenv2 import SnakeEnv2

TOTAL_TIMESTEPS = 100_000
NUM_EVALUATIONS = 5

################## Load Environment ###################

# Goal is 10
env_10 = TransformObservation(SnakeEnv(), lambda obs: obs[:15])

# Goal is 100
env_100 = SnakeEnv()

# Environment 2
env2 = SnakeEnv2()

##################### Load Model ######################

# model = DQN("MlpPolicy", env_100, verbose=0)
model = PPO("MlpPolicy", env2, verbose=0)
# model = A2C("MlpPolicy", env_100, verbose=0)


timesteps_per_batch = int(TOTAL_TIMESTEPS / NUM_EVALUATIONS)

for i in range(1, NUM_EVALUATIONS + 1):
    model.learn(total_timesteps=timesteps_per_batch)
    print(f"\nEVALUATION: {i}/{NUM_EVALUATIONS}\t\tTRAINING STEPS: {timesteps_per_batch * i}\n")
    evaluate_policy(model, SnakeEnv(), n_eval_episodes=10)

model.save("ppo_snake.model")
