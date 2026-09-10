import os
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces


class SO101PickPlaceEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):

        super().__init__()

        self.render_mode = render_mode

        xml_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "assets",
            "scene.xml",
        )

        xml_path = os.path.abspath(xml_path)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # --------------------------------------------------
        # Joint IDs
        # --------------------------------------------------

        self.joint_names = [
            "Shoulder_Rotation",
            "Shoulder_Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
            "Gripper",
        ]

        self.joint_ids = [
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in self.joint_names
        ]

        # --------------------------------------------------
        # Actuator IDs
        # --------------------------------------------------

        self.actuator_ids = [
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
            for name in self.joint_names
        ]

        # --------------------------------------------------
        # Cube
        # --------------------------------------------------

        self.cube_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "cube",
        )

        self.cube_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "freejoint",
        )

        # --------------------------------------------------
        # End effector
        # --------------------------------------------------

        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "Fixed_Jaw",
        )

        # --------------------------------------------------
        # Action
        #
        # 5 arm joints + gripper
        #
        # [-1,1]
        # --------------------------------------------------

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Observation
        #
        # qpos          6
        # qvel          6
        # ee position   3
        # cube position 3
        # target        3
        # cube -> ee    3
        # cube -> goal  3
        # --------------------------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(27,),
            dtype=np.float32,
        )

        # Maximum joint change per environment step
        self.max_joint_delta = np.array([
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.10,
        ])

        self.target = np.array([
            0.15,
            -0.20,
            0.41,
        ])

        self.max_episode_steps = 300
        self.step_count = 0

        self.viewer = None

    # ======================================================
    # RESET
    # ======================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # --------------------------------------------------
        # Robot home
        # --------------------------------------------------

        self.data.qpos[:] = 0

        home = {
            "Shoulder_Rotation": 0.0,
            "Shoulder_Pitch": -1.57,
            "Elbow": 1.5,
            "Wrist_Pitch": 1.22,
            "Wrist_Roll": 0.0,
            "Gripper": 0.25,
        }

        for name, value in home.items():

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            qpos_addr = self.model.jnt_qposadr[joint_id]

            self.data.qpos[qpos_addr] = value

        # --------------------------------------------------
        # Random cube
        # --------------------------------------------------

        cube_x = self.np_random.uniform(
            0.18,
            0.32,
        )

        cube_y = self.np_random.uniform(
            -0.18,
            0.18,
        )

        cube_z = 0.39

        cube_qpos = self.model.jnt_qposadr[
            self.cube_joint_id
        ]

        self.data.qpos[cube_qpos:cube_qpos + 3] = [
            cube_x,
            cube_y,
            cube_z,
        ]

        # Free joint quaternion
        self.data.qpos[cube_qpos + 3:cube_qpos + 7] = [
            1,
            0,
            0,
            0,
        ]

        # --------------------------------------------------
        # Random target
        # --------------------------------------------------

        self.target = np.array([
            self.np_random.uniform(0.10, 0.30),
            self.np_random.uniform(-0.20, 0.20),
            0.41,
        ])

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.step_count = 0

        observation = self._get_obs()

        return observation, {}

    # ======================================================
    # OBSERVATION
    # ======================================================

    def _get_obs(self):

        qpos = np.array([
            self.data.qpos[
                self.model.jnt_qposadr[j]
            ]
            for j in self.joint_ids
        ])

        qvel = np.array([
            self.data.qvel[
                self.model.jnt_dofadr[j]
            ]
            for j in self.joint_ids
        ])

        ee_pos = self.data.xpos[
            self.ee_body_id
        ].copy()

        cube_pos = self.data.xpos[
            self.cube_body_id
        ].copy()

        cube_to_ee = cube_pos - ee_pos

        cube_to_target = self.target - cube_pos

        obs = np.concatenate([
            qpos,
            qvel,
            ee_pos,
            cube_pos,
            self.target,
            cube_to_ee,
            cube_to_target,
        ])

        return obs.astype(np.float32)

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        action = np.clip(
            action,
            -1,
            1,
        )

        # --------------------------------------------------
        # Joint position control
        # --------------------------------------------------

        for i, actuator_id in enumerate(
            self.actuator_ids
        ):

            joint_id = self.joint_ids[i]

            qpos_addr = self.model.jnt_qposadr[
                joint_id
            ]

            current = self.data.qpos[
                qpos_addr
            ]

            target = (
                current
                + action[i]
                * self.max_joint_delta[i]
            )

            # Respect MuJoCo joint limits
            lower = self.model.jnt_range[
                joint_id,
                0,
            ]

            upper = self.model.jnt_range[
                joint_id,
                1,
            ]

            target = np.clip(
                target,
                lower,
                upper,
            )

            self.data.ctrl[
                actuator_id
            ] = target

        # --------------------------------------------------
        # Simulate
        # --------------------------------------------------

        for _ in range(5):

            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.step_count += 1

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        ee_pos = self.data.xpos[
            self.ee_body_id
        ]

        cube_pos = self.data.xpos[
            self.cube_body_id
        ]

        ee_cube_distance = np.linalg.norm(
            ee_pos - cube_pos
        )

        cube_target_distance = np.linalg.norm(
            cube_pos - self.target
        )

        reward = 0.0

        # Reach cube
        reward += -2.0 * ee_cube_distance

        # Move cube toward target
        reward += -2.0 * cube_target_distance

        # Encourage lifting
        if cube_pos[2] > 0.42:

            reward += 2.0

        # Successful placement
        success = (
            cube_target_distance < 0.04
            and cube_pos[2] > 0.40
        )

        if success:

            reward += 100.0

        # Small action penalty
        reward -= 0.001 * np.sum(
            action ** 2
        )

        terminated = success

        truncated = (
            self.step_count
            >= self.max_episode_steps
        )

        observation = self._get_obs()

        info = {
            "success": success,
            "ee_cube_distance": ee_cube_distance,
            "cube_target_distance": cube_target_distance,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ======================================================
    # RENDER
    # ======================================================

    def render(self):

        if self.render_mode != "human":
            return

        if self.viewer is None:

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
            )

        self.viewer.sync()

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        if self.viewer is not None:

            self.viewer.close()

            self.viewer = None