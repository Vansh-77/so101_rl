import time

from envs.so101_pick_place import SO101PickPlaceEnv


env = SO101PickPlaceEnv(
    render_mode="human"
)

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Action shape:", env.action_space.shape)

for episode in range(5):

    obs, info = env.reset()

    done = False

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        env.render()

        time.sleep(0.01)

        done = terminated or truncated

    print(
        f"Episode {episode} | "
        f"reward={reward:.3f} | "
        f"success={info['success']}"
    )

env.close()