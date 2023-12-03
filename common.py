import cv2


def evaluate_policy(model, env, n_eval_episodes: int = 100):

    total_steps = 0
    total_reward = 0
    total_score = 0
    truncated = False

    env.unwrapped.set_render(True)

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

        print(f"{n}:  - Score: {info['score']} - Steps: {episode_steps} "
              f"- Reward: {episode_reward} - Cause of death: {info['cause_of_death']}")
        total_reward += episode_reward
        total_steps += episode_steps
        total_score += info['score']

    print(f"  - Average reward per episode: {total_reward / n_eval_episodes}")
    print(f"  - Average steps per episode: {total_steps / n_eval_episodes}")
    print(f"  - Average score per episode: {total_score / n_eval_episodes}")

    env.unwrapped.set_render(False)
    cv2.destroyAllWindows()
