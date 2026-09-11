import os
import time

from stable_baselines3 import SAC

from envs.so101_pick_place import SO101PickPlaceEnv


MODEL_PATH = "models/so101_sac"


env = SO101PickPlaceEnv(
    render_mode="human"
)

model = SAC.load(
    MODEL_PATH,
    env=env,
)

obs, info = env.reset()

while True:

    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

    if terminated or truncated:

        print(
            f"Episode finished | "
            f"reward={reward:.3f} | "
            f"success={info['success']} | "
            f"grasped={info['grasped']} | "
            f"lifted={info['lifted']}"
        )

        obs, info = env.reset()