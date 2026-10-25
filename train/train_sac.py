import os

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from envs.so101_pick_place import SO101PickPlaceEnv


# ======================================================
# CONFIG
# ======================================================

N_ENVS = 8
TOTAL_STEPS = 500_000

LOG_DIR = "logs/"
MODEL_DIR = "models/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ======================================================
# ENVIRONMENTS
# ======================================================

env = make_vec_env(
    SO101PickPlaceEnv,
    n_envs=N_ENVS,
    env_kwargs={
        "render_mode": "human",
    },
)


# ======================================================
# SAC
# ======================================================

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,

    # Start conservative
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=5_000,

    batch_size=256,

    tau=0.005,
    gamma=0.99,

    train_freq=1,
    gradient_steps=1,

    device="auto",
)

# model = SAC.load(
#     "models/so101_sac",
#     env=env,
# )

# ======================================================
# TRAIN
# ======================================================

model.learn(
    total_timesteps=TOTAL_STEPS,
    progress_bar=True,
)


# ======================================================
# SAVE
# ======================================================

model.save(
    os.path.join(
        MODEL_DIR,
        "so101_sac",
    )
)

env.close()

print("Training complete.")