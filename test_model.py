import sys
import cv2
from gymnasium.wrappers import TransformObservation
from stable_baselines3 import PPO

from snakeenv import SnakeEnv
from snakeenv2 import SnakeEnv2


def evaluate_policy(model, env, n_eval_episodes: int = 100):

    total_steps = 0
    total_reward = 0
    total_score = 0
    truncated = False

    env.unwrapped.activate_render()

    for n in range(n_eval_episodes):

        episode_steps = 0
        episode_reward = 0
        done = False
        info = {}

        obs, _ = env.reset()

        while not done:
            action, _states = model.predict(obs)
            if episode_steps == 0 and action == 0:
                continue
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

        if not truncated:
            info['cause_of_death'] = 'No dead'

        print(f"{n}:\t - Score: {info['score']} - Steps: {episode_steps} "
              f"- Reward: {episode_reward} - Cause of death: {info['cause_of_death']}")
        total_reward += episode_reward
        total_steps += episode_steps
        total_score += info['score']

    print(f"  - Average reward per episode: {total_reward / n_eval_episodes}")
    print(f"  - Average steps per episode: {total_steps / n_eval_episodes}")
    print(f"  - Average score per episode: {total_score / n_eval_episodes}")
    cv2.destroyAllWindows()


try:

    version = int(sys.argv[1])

except:

    print("No parms")
    exit(2)
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
