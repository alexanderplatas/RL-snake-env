import sys
from common import evaluate_policy
from gymnasium.wrappers import TransformObservation
from stable_baselines3 import PPO, DQN, A2C

from snakeenv import SnakeEnv
from snakeenv2 import SnakeEnv2


try:

    version = int(sys.argv[1])

    ##################### Test Model ######################

    if version == 2:

        ppo_model = PPO.load("models/ppo_snake_prueba2.model")
        env_10 = TransformObservation(SnakeEnv(), lambda obs: obs[:15])
        evaluate_policy(ppo_model, env_10, n_eval_episodes=10)

    elif version == 4:

        # Evaluar modelo de la prueba 4
        ppo_model = PPO.load("models/ppo_snake_prueba4.model")
        env_100 = SnakeEnv()
        evaluate_policy(ppo_model, env_100, n_eval_episodes=10)

    elif version == 5:
        # Evaluar modelo del entorno cambiado
        ppo_model = PPO.load("models/ppo_snake_obs2.model")
        env2 = SnakeEnv2()
        evaluate_policy(ppo_model, env2, n_eval_episodes=10)

    else:
        print("Incorrect version")

except KeyboardInterrupt:
    print("  Interrupted")

except Exception as error:
    print("No params")

