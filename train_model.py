from gymnasium.wrappers import TransformObservation
from stable_baselines3 import PPO, DQN, A2C
from common import evaluate_policy
from snakeenv import SnakeEnv
from snakeenv2 import SnakeEnv2

####################### Params ########################

TOTAL_TIMESTEPS = 1_000_000
NUM_EVALUATIONS = 10

################## Load Environment ###################

# Goal is 10
# env = TransformObservation(SnakeEnv(), lambda obs: obs[:15])

# Goal is 100
# env = SnakeEnv()

# Environment 2
env = SnakeEnv2()

##################### Load Model ######################

# model = DQN("MlpPolicy", env, verbose=0)
model = PPO("MlpPolicy", env, verbose=0)
# model = A2C("MlpPolicy", env, verbose=0)

##################### Train Model #####################

timesteps_per_batch = int(TOTAL_TIMESTEPS / NUM_EVALUATIONS)

for i in range(1, NUM_EVALUATIONS + 1):
    model.learn(total_timesteps=timesteps_per_batch)
    print(f"\nEVALUATION: {i}/{NUM_EVALUATIONS}\t\tTRAINING STEPS: {timesteps_per_batch * i}\n")
    evaluate_policy(model, env, n_eval_episodes=5)

##################### Save Model ######################

# model.save("ppo_snake.model")
