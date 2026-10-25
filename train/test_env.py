import tkinter as tk
import time
import numpy as np

from envs.so101_pick_place import SO101PickPlaceEnv


env = SO101PickPlaceEnv(render_mode="human")
obs, info = env.reset()

JOINT_NAMES = env.joint_names
JOINT_IDS = env.joint_ids
ACTUATOR_IDS = env.actuator_ids

root = tk.Tk()
root.title("SO-101 Manual Control")
root.geometry("440x540")

sliders = []
target_positions = np.array(
    [env.data.qpos[jid] for jid in JOINT_IDS],
    dtype=np.float64,
)

joint_limits = [
    env.model.jnt_range[jid].copy()
    for jid in JOINT_IDS
]

for i, name in enumerate(JOINT_NAMES):
    low, high = joint_limits[i]

    frame = tk.Frame(root)
    frame.pack(fill="x", padx=12, pady=6)

    label = tk.Label(frame, text=name, width=18, anchor="w")
    label.pack(side="left")

    slider = tk.Scale(
        frame,
        from_=float(low),
        to=float(high),
        resolution=0.01,
        orient="horizontal",
        length=240,
        command=lambda value, idx=i: set_target(idx, value),
    )
    slider.set(target_positions[i])
    slider.pack(side="right")

    sliders.append(slider)


def set_target(index, value):
    target_positions[index] = float(value)


status = tk.Label(
    root,
    text="Starting...",
    justify="left",
    anchor="w",
    font=("TkDefaultFont", 10),
)
status.pack(fill="x", padx=12, pady=12)


def update():
    global obs, info

    try:
        # Convert desired joint positions to the incremental actions
        # expected by this environment's step() method.
        current_positions = np.array(
            [env.data.qpos[jid] for jid in JOINT_IDS],
            dtype=np.float64,
        )

        max_delta = np.asarray(env.max_joint_delta, dtype=np.float64)

        action = np.clip(
            (target_positions - current_positions) / max_delta,
            -1.0,
            1.0,
        )

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        status.config(
            text=(
                f"Reward: {reward:.3f}\n"
                f"Stage: {info.get('grasp_stage', 'N/A')}\n"
                f"Grasped: {info.get('grasped', False)}\n"
                f"Lifted: {info.get('lifted', False)}\n"
                f"Success: {info.get('success', False)}\n"
                f"Cube height: {info.get('cube_height', 0):.3f} m"
            )
        )

        # Do not reset automatically: preserve the current scene
        # so you can inspect failures and continue manually.
        if terminated or truncated:
            status.config(
                text=status.cget("text")
                + "\nEpisode ended. Press Reset to start again."
            )
            root.after_cancel(update_job)
            return

        root.after(20, update)

    except tk.TclError:
        pass


def reset_environment():
    global obs, info, target_positions

    obs, info = env.reset()

    target_positions = np.array(
        [env.data.qpos[jid] for jid in JOINT_IDS],
        dtype=np.float64,
    )

    for i, slider in enumerate(sliders):
        slider.set(target_positions[i])

    update()


reset_button = tk.Button(
    root,
    text="Reset Environment",
    command=reset_environment,
)
reset_button.pack(pady=8)

try:
    update_job = root.after(20, update)
    root.mainloop()

finally:
    env.close()